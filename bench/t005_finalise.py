"""Finalise one T005 directory written by bench/opt_ladder.py (a ladder, a Part A pair or an E4i search):
equivalence.csv from raw/actions/*.npz, summary.md from ladder.csv, meta.json (protocol §5) with every raw file hashed,
compressed above 1 MB and secrets-scanned.

Equivalence (task T005 step 3): every rung's actions on the same 10 seeded-noise inferences (torch.manual_seed(5000+i)
before each, the fixture input) against two references in the same directory:
  eager_fp32                  the baseline
  eager_bf16 (LeRobot) / eager_tf32 (GR00T, already all-bf16)   the eager bf16 path
max |Δ| and relative ‖Δ‖₂ over all 10 chunks; recorded, never tuned. References may come from another directory
(--ref-dir), e.g. an E4i search compared with its model's ladder.

python bench/t005_finalise.py --run results/<dir> --bench-commit <sha> --command "<cmd>" [--ref-dir results/<ladder>]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_rows(run):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{run}/raw/rows/*.json"))]


def equivalence(run, ref_dirs):
    acts = {}
    for d in [run] + ref_dirs:
        for f in glob.glob(f"{d}/raw/actions/*.npz"):
            key = os.path.basename(f)[:-4]
            acts.setdefault(key if d == run else f"{os.path.basename(d.rstrip('/'))}:{key}", np.load(f)["actions"])
    out = []
    for key, a in sorted(acts.items()):
        if ":" in key:
            continue
        n = key.rsplit("_N", 1)[1]
        for ref in ("eager_fp32", "eager_bf16", "eager_tf32"):
            cands = [k for k in acts if k.split(":")[-1] == f"{ref}_N{n}"]
            if not cands:
                continue
            rk = f"{ref}_N{n}" if f"{ref}_N{n}" in acts else cands[0]
            r = acts[rk]
            if r.shape != a.shape:
                out.append({"rung_N": key, "reference": rk, "max_abs": "n/a: shape differs", "rel_l2": "", "finite": ""})
                continue
            d = a.astype(np.float64) - r.astype(np.float64)
            out.append({"rung_N": key, "reference": rk, "max_abs": f"{np.abs(d).max():.4e}",
                        "rel_l2": f"{np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30):.4e}",
                        "ref_max_abs_value": f"{np.abs(r).max():.4e}",
                        "finite": bool(np.isfinite(a).all()), "chunks": a.shape[0], "values_per_chunk": a.shape[1]})
    if out:
        keys = []
        for r in out:
            keys += [k for k in r if k not in keys]
        with open(f"{run}/equivalence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out)
    return out


def fmt(r, k):
    v = r.get(f"{k}_median_ms", "")
    if v == "":
        return "—"
    return f"{float(v):.2f} ({float(r[f'{k}_p5_ms']):.2f}–{float(r[f'{k}_p95_ms']):.2f})"


def summary(run, eq):
    rows = list(csv.DictReader(open(f"{run}/ladder.csv")))
    raw = {(d["rung"], str(d["N"])): d for d in load_rows(run)}
    model = rows[0]["model"] if rows else "?"
    groot = model.startswith("groot")
    L = [f"# {os.path.basename(run.rstrip('/'))} — T005, {model}, MAXN, jetson_clocks, gate passed per rung", "",
         f"Median (p5–p95) ms over n = {'/'.join(sorted({r.get('e2e_wall_n', '') for r in rows if r.get('e2e_wall_n')}))} timed inferences per rung, each rung in its own process; regions installed "
         "after compilation. `other` = e2e − vision − prefix − expert per iteration."
         + (" GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e." if groot else ""), ""]
    for n in sorted({r["N"] for r in rows}, key=int):
        L += [f"## N = {n}", ""]
        hdr = "| rung | status | e2e | vision | prefix | expert | other |" + (" model call | AE % of model call |" if groot else " AE % |")
        hdr += " ms/step (instr.) | first call s | graphs in timed | thr. | state |"
        L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for r in [r for r in rows if r["N"] == n]:
            if r["status"] != "measured":
                L.append(f"| `{r['rung']}` | {r['status']}: {r['reason_or_error'][:160]} |" + " |" * (hdr.count("|") - 3))
                continue
            share = r.get("ae_share_of_model_call" if groot else "ae_share_of_e2e", "")
            st = r.get("ae_step_mean_of_medians_ms", "")
            try:
                st = f"{float(st):.2f}"
            except ValueError:
                st = "—"
            L.append(f"| `{r['rung']}` | measured | {fmt(r, 'e2e_wall')} | {fmt(r, 'vision')} | {fmt(r, 'vlm_prefix')} | "
                     f"{fmt(r, 'ae_total')} | {fmt(r, 'other')} |"
                     + (f" {fmt(r, 'model_get_action')} |" if groot else "")
                     + (f" {float(share) * 100:.1f} |" if share else " — |")
                     + f" {st} | {r['first_call_s']} | {r['graphs_compiled_during_timed']} | {r['throttled']} | {r['memory_state']} |")
        slopes = []
        for (rung, N), d in raw.items():
            if N != n or not d.get("by_N") or len(d["by_N"]) < 3:
                continue
            pts = sorted((int(k), v["regions"]["ae_total" if "ae_total" in v["regions"] else "e2e_wall"]["median_ms"])
                         for k, v in d["by_N"].items())
            k = len(pts)
            mx, my = sum(p[0] for p in pts) / k, sum(p[1] for p in pts) / k
            b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
            which = "ae_total" if "ae_total" in d["by_N"][n]["regions"] else "e2e"
            slopes.append(f"`{rung}`: {b:.2f} ms/step ({which}, N = {[p[0] for p in pts]}, intercept {my - b * mx:.2f})")
        if slopes:
            L += ["", "Per-step time by slope: " + "; ".join(sorted(slopes)) + "."]
        L.append("")
    if eq:
        L += ["## Equivalence (10 seeded-noise inferences; `equivalence.csv`)", "",
              "| rung | reference | max abs Δ | relative L2 Δ | finite |", "|---|---|---|---|---|"]
        L += [f"| `{e['rung_N']}` | `{e['reference']}` | {e['max_abs']} | {e['rel_l2']} | {e['finite']} |" for e in eq]
        L.append("")
    preps = sorted({(r["rung"], r["preps"]) for r in rows if r.get("preps")})
    if preps:
        L += ["## Preparations", ""] + [f"- `{k}`: {v}" for k, v in preps] + [""]
    open(f"{run}/summary.md", "w").write("\n".join(L) + "\n")
    return rows, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench-commit", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--experiment", default="E4 Part B optimisation ladder")
    ap.add_argument("--ref-dir", action="append", default=[])
    ap.add_argument("--python", default=os.path.expanduser("~/venvs/base/bin/python"))
    a = ap.parse_args()
    R = a.run.rstrip("/")
    subprocess.run([sys.executable, os.path.join(HERE, "opt_ladder.py"), "--model", "x", "--out", R, "--merge-only",
                    "--n", "1"], check=True)
    eq = equivalence(R, a.ref_dir)
    rows, raw = summary(R, eq)
    d0 = next(iter(raw.values()))
    prov = next((d["provenance"] for d in raw.values() if d.get("provenance")), {})
    ns = sorted({str(d["N"]) for d in raw.values()}, key=int)
    tok = next((d["tokens"] for d in raw.values() if d.get("tokens")), {})
    fields = {
        "experiment": a.experiment, "task": "T005", "model": d0["model"],
        "model_repo": prov.get("model_repo", "n/a"), "model_commit": prov.get("model_commit", "n/a"),
        "checkpoint": prov.get("checkpoint", "n/a"), "checkpoint_revision": prov.get("checkpoint_revision", "n/a"),
        "precision": "per rung: raw/rows/<rung>_N<N>.json rung_desc.precision_flags and param_dtype_bytes",
        "rung": "per row: " + ", ".join(sorted({r["rung"] for r in rows})),
        "compile_mode": "per rung: rung_desc.compiled (bench/rungs.py)", "variant": "n/a: ladder rungs as defined in bench/rungs.py",
        "num_denoise_steps": ", ".join(ns) + " (extra N per rung for the slope: by_N in the row files)",
        "action_horizon": prov.get("action_horizon", {"groot_n17": 40, "groot_n16": 16, "groot_n15": 16}.get(d0["model"], "n/a")),
        "prefix_tokens": tok.get("prefix_tokens_P", "n/a"), "suffix_tokens": tok.get("suffix_tokens_S", "n/a"),
        "batch": 1, "n_warmup": ">= 20 and until no new graph for 5 inferences (warmup_iters per row)", "n_timed": 100,
        "hw_constants_source": "results/20261022_E1_peak/hw_constants.json", "bench_commit": a.bench_commit,
        "command": a.command,
        "throttled": {f"{d['rung']}_N{d['N']}": {str(n): b["throttle"]["throttled"] for n, b in (d.get("by_N") or {}).items()}
                      for d in raw.values()},
        "memory_path_state": {f"{d['rung']}_N{d['N']}": {str(n): f"{b['memory_state_before']['state']}->{b['memory_state_after']['state']}"
                                                          for n, b in (d.get("by_N") or {}).items()} for d in raw.values()},
        "produced_by": {"_N": "bench/opt_ladder.py (child)", ".log": "bench/opt_ladder.py child stdout with TORCH_LOGS=recompiles,graph_breaks",
                        "traceback_": "bench/opt_ladder.py", "secrets_scan": "bench/secrets_scan.py"},
        "also_raw": [],
        "nsys_version": "n/a", "ncu_version": "n/a", "ncu_set": "n/a", "ncu_filter": "n/a", "ncu_metrics": "n/a",
        "cuobjdump_version": "n/a", "ncu_sm_frequency_hz": "n/a",
        "notes": "One process per rung; regions after compilation (bench/regions.py); warm-up until torch._dynamo's "
                 "unique_graphs counter is stable; graphs_compiled_during_timed must be 0. Every timed block began in the "
                 "normal memory-path state (T003). Equivalence in equivalence.csv."}
    fp = f"{R}/raw/meta_fields.json"
    json.dump(fields, open(fp, "w"), indent=1, default=str)
    scan = f"{R}/raw/secrets_scan.csv"
    files = [p for p in glob.glob(f"{R}/**", recursive=True) if os.path.isfile(p)]
    subprocess.run([a.python, os.path.join(HERE, "secrets_scan.py"), "--out", scan, "--token-file",
                    os.path.expanduser("~/.cache/huggingface/token")] + files, check=False)
    subprocess.run([a.python, os.path.join(HERE, "meta.py"), "--run", R, "--fields", fp, "--device-state", "--scan", scan],
                   check=True)
    print(open(f"{R}/summary.md").read())


if __name__ == "__main__":
    main()
