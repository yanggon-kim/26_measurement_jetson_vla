"""T005 / E4 Part A, Part B (the ladder) and E4i (the search): one rung = one fresh process, per-region latency.

Ported from reference/orin/bench/opt_ladder.py and groot_phases.py onto T004's harness (bench/run_e3.py adapters,
bench/regions.py), so eager_fp32 here is measured exactly as E3 measured it. The rungs themselves are bench/rungs.py.

Per rung (child process, --single):
  gate (protocol §1, GPU idle) -> load a fresh model -> apply_rung (precision, preps, compile) -> instrument regions
  AFTER compiling (every region boundary outside a compiled graph) -> warm-up >= --warmup inferences and until no new
  graph has been compiled for 5 consecutive inferences (torch._dynamo counters["stats"]["unique_graphs"]) -> the
  memory-path state is waited into "normal" (T003) -> --timed inferences, CUDA events per region every iteration ->
  checks that no graph was compiled during the timed loop -> optionally --save-actions K inferences with
  torch.manual_seed(5000 + i) for the equivalence check -> raw/rows/<rung>_N<N>.json + raw/samples/<rung>_N<N>.csv.
  Extra N values (--slope-steps) repeat warm-up + timing at those N in the same process, for the per-step time by
  slope where the per-step regions are inside a compiled graph (GR00T's compiled head, whole_graphs).
The parent (--rungs) runs the children in sequence, logs each child's stdout with TORCH_LOGS=recompiles,graph_breaks
into raw/logs/, gives each its own Inductor / Triton cache, and merges the rows into ladder.csv.

Regions: as E3 (vision, vlm_prefix, ae_total, ae_step[i]; GR00T also model_get_action, vlm_backbone,
backbone_lm_head, outside_model). "other" = e2e − vision − vlm_prefix − ae_total per iteration. GR00T's e2e is
Gr00tPolicy.get_action; model_get_action (the Orin's GR00T e2e) is reported beside it.

python bench/opt_ladder.py --model pi0 --out results/<date>_E4c_pi0_opt_ladder --rungs eager_fp32,eager_tf32,...
python bench/opt_ladder.py --model pi0 --out <dir> --rung best_bf16      # resolves results/*_E4i_summary/best_bf16.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def summarize(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]  # noqa: E731
    return {"median_ms": statistics.median(v), "mean_ms": statistics.mean(v), "p5_ms": q(.05), "p95_ms": q(.95),
            "min_ms": v[0], "max_ms": v[-1], "n": len(v)}


def resolve_best(model):
    """--rung best_bf16: the rung key T005 named for this model (the newest results/*_E4i_summary/best_bf16.json)."""
    fs = sorted(glob.glob(os.path.join(REPO, "results", "*_E4i_summary", "best_bf16.json")))
    if not fs:
        raise SystemExit("no results/*_E4i_summary/best_bf16.json yet (T005 step 4)")
    d = json.load(open(fs[-1]))
    if model not in d["models"]:
        raise SystemExit(f"{fs[-1]} names no best_bf16 for {model}")
    return d["models"][model]["rung"], fs[-1]


def flatten_actions(obj):
    """Every tensor / ndarray in an action structure, flattened in a fixed key order, as one float32 vector."""
    import numpy as np
    import torch
    out = []

    def walk(x):
        if torch.is_tensor(x):
            out.append(x.detach().float().cpu().numpy().ravel())
        elif isinstance(x, np.ndarray):
            out.append(x.astype(np.float32).ravel())
        elif isinstance(x, dict) or hasattr(x, "keys"):
            for k in sorted(x.keys()):
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
    walk(obj)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ------------------------------------------------------------------------------------------------ child
def child(a):
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    import gate as G
    import regions as R
    import rungs as RG
    import run_e3 as E3

    key = a.single
    out = a.out
    os.makedirs(os.path.join(out, "raw", "rows"), exist_ok=True)
    os.makedirs(os.path.join(out, "raw", "samples"), exist_ok=True)
    tag = f"{key}_N{a.n}"
    row = {"model": a.model, "rung": key, "N": a.n, "status": "failed", "error": "", "not_applicable_reason": ""}
    t_start = time.time()
    try:
        g0 = G.wait_for_gate(apply_clocks=True)
        row["gate_start"] = {k: g0[k] for k in ("time_utc", "checks", "gpu_hz", "gpu_util_pct_samples",
                                                "soctherm_oc_event_cnt", "thermal_zones_mC")}
        ad = E3.make_adapter(a.model)
        ad.name, ad.family = a.model, RG.family_of(a.model)
        row["provenance"] = ad.provenance
        if ad.family == "groot":
            ad.capture_tokens()
        row["tokens"] = {k: v for k, v in ad.tokens.items() if k != "shapes"}
        try:
            desc = RG.apply_rung(ad, key)
        except RG.NotApplicable as e:
            row.update(status="not applicable", not_applicable_reason=str(e))
            json.dump(row, open(os.path.join(out, "raw", "rows", tag + ".json"), "w"), indent=1, default=str)
            print(f"{tag}: NOT APPLICABLE: {e}", flush=True)
            return 0
        row["rung_desc"] = desc
        row["param_dtype_bytes"] = {}
        for p in ad.policy.model.parameters():
            row["param_dtype_bytes"][str(p.dtype)] = row["param_dtype_bytes"].get(str(p.dtype), 0) + p.numel() * p.element_size()
        rec = R.Recorder()
        undo = None
        if desc["regions"]:
            if ad.family == "groot":
                undo, _ = R.instrument_groot(ad.policy, rec, inner=not desc["ae_compiled"])
            else:
                undo = R.instrument_lerobot(ad.policy, rec, ad.spec.container)
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

        def one():
            mark()
            return ad.iteration()

        def ugraphs():
            return int(counters["stats"].get("unique_graphs", 0))

        row["by_N"] = {}
        steps = [a.n] + [n for n in a.slope_steps if n != a.n]
        for n in steps:
            ad.set_n(n)
            t0 = time.perf_counter()
            one()
            torch.cuda.synchronize()
            first_s = time.perf_counter() - t0
            hist, it = [ugraphs()], 1
            while it < a.warmup or (len(hist) > 5 and hist[-1] != hist[-6]) or len(hist) <= 5:
                rec.reset()
                one()
                hist.append(ugraphs())
                it += 1
                if it >= a.warmup_max:
                    break
            torch.cuda.synchronize()
            warm_stable = len(hist) > 5 and hist[-1] == hist[-6]
            ms0 = G.memory_state_probe()
            for _ in range(6):
                if ms0["state"] == "normal":
                    break
                time.sleep(30)
                ms0 = G.memory_state_probe()
            torch.cuda.reset_peak_memory_stats()
            ug0, oc0, hz = ugraphs(), G.oc_counts(), []
            rows = []
            t_loop = time.perf_counter()
            for i in range(a.timed):
                rec.reset()
                rec.start()
                walls, _ = one()
                rec.stop()
                r = rec.results_ms(ad.derive_ae_total) if desc["regions"] else {}
                r.update(walls)
                if "model_get_action" in r:
                    r["outside_model"] = r["e2e_wall"] - r["model_get_action"]
                if all(k in r for k in ("vision", "vlm_prefix", "ae_total")):
                    r["other"] = r["e2e_wall"] - r["vision"] - r["vlm_prefix"] - r["ae_total"]
                for k, v in r.items():
                    rows.append({"N": n, "region": k, "iter": i, "ms": v})
                if desc["regions"]:
                    for j, v in enumerate(rec.step_times_ms()):
                        rows.append({"N": n, "region": f"ae_step_individual[{j}]", "iter": i, "ms": v})
                if i % 10 == 0:
                    hz.append(int(open(f"{G.DEVFREQ}/cur_freq").read()))
            loop_s = time.perf_counter() - t_loop
            oc1, ug1 = G.oc_counts(), ugraphs()
            by = {}
            for x in rows:
                by.setdefault(x["region"], []).append(x["ms"])
            st = {k: summarize(v) for k, v in by.items()}
            steps_med = [st[f"ae_step_individual[{j}]"]["median_ms"] for j in range(n) if f"ae_step_individual[{j}]" in st]
            row["by_N"][n] = {
                "regions": st, "first_call_s": first_s, "warmup_iters": it, "warmup_unique_graphs": hist,
                "warmup_stable": warm_stable, "graphs_compiled_during_timed": ug1 - ug0,
                "ae_step_mean_of_medians_ms": (sum(steps_med) / len(steps_med)) if steps_med else "n/a: per-step regions off",
                "peak_mem_allocated_bytes": torch.cuda.max_memory_allocated(),
                "control_frequency_hz": a.timed / loop_s, "throttle": G.throttle_verdict(oc0, oc1, hz),
                "memory_state_before": ms0, "memory_state_after": G.memory_state_probe()}
            with open(os.path.join(out, "raw", "samples", f"{key}_N{n}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["N", "region", "iter", "ms"])
                w.writeheader()
                w.writerows(rows)
            e = st["e2e_wall"]["median_ms"]
            print(f"{key} N={n}: e2e {e:.2f} | vision {st.get('vision', {}).get('median_ms', float('nan')):.2f} | prefix "
                  f"{st.get('vlm_prefix', {}).get('median_ms', float('nan')):.2f} | ae {st.get('ae_total', {}).get('median_ms', float('nan')):.2f}"
                  f" | graphs during timed {ug1 - ug0} | throttled {row['by_N'][n]['throttle']['throttled']} | state "
                  f"{ms0['state']}->{row['by_N'][n]['memory_state_after']['state']}", flush=True)
        if a.save_actions:
            ad.set_n(a.n)
            acts = []
            for i in range(a.save_actions):
                torch.manual_seed(5000 + i)
                _, act = one()
                acts.append(flatten_actions(act))
            A = np.stack(acts)
            os.makedirs(os.path.join(out, "raw", "actions"), exist_ok=True)
            np.savez(os.path.join(out, "raw", "actions", f"{tag}.npz"), actions=A, seeds=np.arange(5000, 5000 + a.save_actions))
            row["actions_finite"] = bool(np.isfinite(A).all())
            row["actions_shape"] = list(A.shape)
        row["inductor_counters"] = {k: dict(v) for k, v in counters.items() if k in ("inductor", "stats", "graph_break")}
        row["mpad_table"] = getattr(ad, "mpad_table", "n/a: linear_mpad not applied")
        if undo:
            undo()
        row["status"] = "measured"
    except Exception as e:                                                          # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        open(os.path.join(out, "raw", f"traceback_{tag}.txt"), "w").write(traceback.format_exc())
        print(f"{tag}: FAILED {row['error']}", flush=True)
    row["wall_s"] = time.time() - t_start
    json.dump(row, open(os.path.join(out, "raw", "rows", tag + ".json"), "w"), indent=1, default=str)
    return 0 if row["status"] != "failed" else 3


# ------------------------------------------------------------------------------------------------ parent
def parent(a):
    import rungs as RG
    keys = a.rungs.split(",")
    os.makedirs(os.path.join(a.out, "raw", "logs"), exist_ok=True)
    cache_root = os.path.join(REPO, "scratch", "t005", "cache", os.path.basename(a.out.rstrip("/")))
    best_src = {}
    for key in keys:
        real = key
        if key == "best_bf16":
            real, src = resolve_best(a.model)
            best_src = {"best_bf16": real, "source": os.path.relpath(src, REPO)}
        RG.get(a.model, real)                     # fail early on a typo
        tag = f"{real}_N{a.n}"
        if os.path.exists(os.path.join(a.out, "raw", "rows", tag + ".json")) and not a.redo:
            print(f"=== {tag}: row exists, skipped", flush=True)
            continue
        slope = a.slope_steps if (a.slope_rungs == "all" or real in a.slope_rungs.split(",")) else ""
        cmd = [sys.executable, os.path.abspath(__file__), "--model", a.model, "--out", os.path.abspath(a.out),
               "--single", real, "--n", str(a.n), "--warmup", str(a.warmup), "--warmup-max", str(a.warmup_max),
               "--timed", str(a.timed), "--save-actions", str(a.save_actions), "--slope-steps", slope]
        env = dict(os.environ, TORCH_LOGS="recompiles,graph_breaks",
                   TORCHINDUCTOR_CACHE_DIR=os.path.join(cache_root, real, "inductor"),
                   TRITON_CACHE_DIR=os.path.join(cache_root, real, "triton"))
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            env.pop(k, None)
        env["HF_HUB_OFFLINE"] = "1"
        print(f"=== {tag}: {time.strftime('%H:%M:%S')} start", flush=True)
        t0 = time.time()
        with open(os.path.join(a.out, "raw", "logs", tag + ".log"), "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                cwd="/tmp" if a.model.startswith("groot") else REPO).returncode
        log = open(os.path.join(a.out, "raw", "logs", tag + ".log")).read()
        last = [ln for ln in log.splitlines() if ln.startswith(real) or "FAILED" in ln or "NOT APPLICABLE" in ln][-3:]
        print(f"=== {tag}: exit {rc} in {time.time() - t0:.0f} s | " + " || ".join(last), flush=True)
        time.sleep(a.pause)
    merge(a.out, best_src)


def merge(out, best_src=None):
    """raw/rows/*.json -> ladder.csv (one row per rung and N, every part's median / p5 / p95 / n)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(out, "raw", "rows", "*.json"))):
        d = json.load(open(f))
        for n, b in (d.get("by_N") or {"": {}}).items():
            r = {"model": d["model"], "rung": d["rung"], "N": n or d["N"], "status": d["status"],
                 "primary_N": str(n) == str(d["N"]) or not n,
                 "reason_or_error": d.get("not_applicable_reason") or d.get("error", ""),
                 "compiled": ";".join((d.get("rung_desc") or {}).get("compiled", [])),
                 "preps": " | ".join((d.get("rung_desc") or {}).get("preps", [])),
                 "autotune": (d.get("rung_desc") or {}).get("autotune", ""),
                 "matmul_precision": ((d.get("rung_desc") or {}).get("precision_flags") or {}).get("float32_matmul_precision", "")}
            st = b.get("regions", {})
            for k in ("e2e_wall", "vision", "vlm_prefix", "ae_total", "other", "model_get_action", "outside_model",
                      "vlm_backbone", "backbone_lm_head"):
                if k in st:
                    for q in ("median_ms", "p5_ms", "p95_ms"):
                        r[f"{k}_{q}"] = round(st[k][q], 4)
                    r[f"{k}_n"] = st[k]["n"]
            if b:
                r.update({"ae_step_mean_of_medians_ms": b.get("ae_step_mean_of_medians_ms"),
                          "first_call_s": round(b["first_call_s"], 2), "warmup_iters": b["warmup_iters"],
                          "graphs_compiled_during_timed": b["graphs_compiled_during_timed"],
                          "peak_mem_GB": round(b["peak_mem_allocated_bytes"] / 1e9, 3),
                          "throttled": b["throttle"]["throttled"],
                          "memory_state": f"{b['memory_state_before']['state']}->{b['memory_state_after']['state']}",
                          "actions_finite": d.get("actions_finite", "")})
                if "e2e_wall" in st and "ae_total" in st:
                    r["ae_share_of_e2e"] = round(st["ae_total"]["median_ms"] / st["e2e_wall"]["median_ms"], 4)
                if "model_get_action" in st and "ae_total" in st:
                    r["ae_share_of_model_call"] = round(st["ae_total"]["median_ms"] / st["model_get_action"]["median_ms"], 4)
            rows.append(r)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, "ladder.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    if best_src:
        json.dump(best_src, open(os.path.join(out, "best_bf16_resolved.json"), "w"), indent=1)
    print(f"merged {len(rows)} rows -> {out}/ladder.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default="", help="comma list for the parent")
    ap.add_argument("--rung", default="", help="one rung (e.g. best_bf16); same as --rungs with one entry")
    ap.add_argument("--single", default="", help="internal: run this rung in this process")
    ap.add_argument("--n", type=int, default=0, help="N (default: the model's own)")
    ap.add_argument("--slope-steps", default="", help="extra N for the per-step slope, e.g. 2,6")
    ap.add_argument("--slope-rungs", default="", help="rungs that get --slope-steps ('all' or a comma list)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--warmup-max", type=int, default=80)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--save-actions", type=int, default=10)
    ap.add_argument("--pause", type=int, default=20, help="seconds between children (memory reclaim, cool-down)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    if not a.n:
        import model_registry as MR
        a.n = MR.GROOT_VERSIONS[a.model]["n_default"] if a.model.startswith("groot") else MR.get(a.model).n_default
    a.slope_steps = [int(x) for x in a.slope_steps.split(",") if x] if a.single else a.slope_steps
    if a.merge_only:
        merge(a.out)
        return 0
    if a.single:
        return child(a)
    a.rungs = a.rungs or a.rung
    return parent(a)


if __name__ == "__main__":
    raise SystemExit(main())
