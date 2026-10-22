"""Finalise one E3 result directory (T004): summary.md from its CSVs, meta.json (protocol §5) from raw/run_info.json,
raw/per_N_*.json and the device state, with every raw file hashed and scanned.

python bench/e3_finalise.py --run results/<date>_E3_<model>_eager_Nsweep --bench-commit <sha> --command "<cmd>"
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench-commit", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    R = a.run
    info = json.load(open(f"{R}/raw/run_info.json"))
    per_n = {int(os.path.basename(p)[6:-5]): json.load(open(p)) for p in glob.glob(f"{R}/raw/per_N_*.json")}
    summ = {}
    for r in csv.DictReader(open(f"{R}/latency_summary.csv")):
        summ.setdefault(int(r["N"]), {})[r["region"]] = r
    n0 = info["n_default"]
    s = summ[n0]
    fits = {r["region"]: r for r in csv.DictReader(open(f"{R}/step_fit.csv"))}
    floors = [r for r in csv.DictReader(open(f"{R}/floor.csv")) if r["peak_source"].startswith("measured")]
    g = lambda k: float(s[k]["median_ms"]) if k in s else None  # noqa: E731
    e2e = g("e2e_wall")
    parts = [(k, g(k)) for k in ("vision", "vlm_prefix", "ae_total") if g(k) is not None]
    other = e2e - sum(v for _, v in parts)
    lines = [f"# E3 {info['model']} eager N sweep (T004) — {info['precision_mode']}, MAXN, jetson_clocks, gate passed per N",
             "", f"At the model's default N = {n0} (median of {s['e2e_wall']['n']} timed iterations, CUDA events per region):", "",
             "| region | median ms | p5–p95 ms | share of e2e |", "|---|---|---|---|"]
    for k, v in parts + [("other (e2e − named regions)", other), ("e2e (model call)", e2e)]:
        if k in s:
            lines.append(f"| {k} | {v:.2f} | {float(s[k]['p5_ms']):.2f}–{float(s[k]['p95_ms']):.2f} | {v / e2e * 100:.1f} % |")
        else:
            lines.append(f"| {k} | {v:.2f} | — | {v / e2e * 100:.1f} % |")
    for k in ("model_get_action", "outside_model", "preprocess_wall", "postprocess_wall", "e2e_full_wall", "vlm_backbone",
              "backbone_lm_head"):
        if k in s:
            lines.append(f"| {k} | {g(k):.2f} | {float(s[k]['p5_ms']):.2f}–{float(s[k]['p95_ms']):.2f} | |")
    fa = fits.get("ae_total")
    st = [g(f"ae_step_individual[{j}]") for j in range(n0) if g(f"ae_step_individual[{j}]") is not None]
    lines += ["", f"ms per denoising step: mean of the per-step medians {sum(st) / len(st):.2f} (step 0 {st[0]:.2f}, last {st[-1]:.2f}); "
              f"slope over N {fa['b_ms_per_step'] if fa else 'n/a'} (r² {fa['r2'] if fa else 'n/a'}, ae_total = a + b·N, a = {fa['a_ms'] if fa else 'n/a'} ms)."]
    lines += ["", "| part | floor ms (measured peaks) | bound | measured ms | gap ratio |", "|---|---|---|---|---|"]
    # restream is compared with the sum of the per-step medians (it counts the steps' bytes only; GR00T's ae_total
    # also holds a prologue -- state encoder, vl layer norm -- before the first step, T004 review)
    meas = {"vision": g("vision"), "vlm_prefix": g("vlm_prefix"), "ae_step": float(fa["b_ms_per_step"]) if fa else sum(st) / len(st),
            "ae_total": g("ae_total"), "ae_total_restream": sum(st)}
    for f in floors:
        mval = meas.get(f["region"])
        gap = f"{mval / float(f['floor_ms']):.2f}" if mval and float(f["floor_ms"]) else "n/a"
        lines.append(f"| {f['region']} | {float(f['floor_ms']):.3f} | {f['bound']} ({f['dominant_param_dtype']}) | "
                     f"{mval:.2f} | {gap} |" if mval else f"| {f['region']} | {float(f['floor_ms']):.3f} | {f['bound']} | — | — |")
    thr = {n: d["throttle"]["throttled"] for n, d in per_n.items()}
    ms = {n: f"{d['memory_state_before']['state']}->{d['memory_state_after']['state']}" for n, d in per_n.items()}
    lines += ["", f"Tokens: {json.dumps({k: v for k, v in info['tokens'].items() if k != 'shapes'})}.",
              f"Throttled per N: {thr}. Memory-path state per N: {ms}.",
              f"Peak memory at N = {n0}: {per_n[n0]['peak_mem_allocated_bytes'] / 1e9:.2f} GB allocated; control frequency "
              f"{per_n[n0]['control_frequency_hz']:.2f} Hz (timed loop)."]
    open(f"{R}/summary.md", "w").write("\n".join(lines) + "\n")
    prov = info["provenance"]
    fields = {
        "experiment": "E3 eager end-to-end latency, N sweep", "task": "T004 step 4", "model": info["model"],
        "model_repo": prov["model_repo"], "model_commit": prov["model_commit"], "checkpoint": prov["checkpoint"],
        "checkpoint_revision": prov["checkpoint_revision"],
        "precision": f"{info['precision_mode']}: flags after load {info['precision_flags_after_load']}; param dtypes {info['param_dtypes']}",
        "rung": "eager_fp32" if info["precision_mode"] == "eager_fp32" else "shipped eager path",
        "compile_mode": "none (eager)", "variant": "n/a", "num_denoise_steps": f"sweep {info['steps']} (default {n0})",
        "action_horizon": prov.get("action_horizon", {"groot_n17": 40, "groot_n16": 16, "groot_n15": 16}.get(info["model"], "n/a")),
        "prefix_tokens": info["tokens"].get("prefix_tokens_P"), "suffix_tokens": info["tokens"].get("suffix_tokens_S"),
        "batch": 1, "n_warmup": info["warmup"], "n_timed": info["timed"],
        "hw_constants_source": "results/20261022_E1_peak/hw_constants.json", "bench_commit": a.bench_commit,
        "command": a.command,
        "throttled": {str(n): d["throttle"] for n, d in per_n.items()},
        "memory_path_state": {str(n): [d["memory_state_before"], d["memory_state_after"]] for n, d in per_n.items()},
        "provenance": prov,
        "produced_by": {"per_N": "bench/run_e3.py", "run_info": "bench/run_e3.py", "gate_start": "bench/gate.py",
                        "run_e3": "bench/run_e3.py (stdout)", "meta_fields": "bench/e3_finalise.py",
                        "secrets_scan": "bench/secrets_scan.py", "latency_samples": "bench/run_e3.py"},
        "also_raw": ["latency_samples.csv"],
        "notes": "Regions: bench/regions.py; e2e = the model call (LeRobot: predict_action_chunk; GR00T: Gr00tPolicy.get_action). "
                 "Floors (floor.csv) from one accounting pass at the default N: FLOPs by torch FlopCounterMode, bytes = "
                 "parameters touched per region (embeddings: rows looked up) + the hand-over (accounting.json); ceilings "
                 "for the dominant parameter dtype of the part. Every N block started in the normal memory-path state."}
    def no_null(x):
        if isinstance(x, dict):
            return {k: no_null(v) for k, v in x.items()}
        if isinstance(x, list):
            return [no_null(v) for v in x]
        return "n/a: not applicable to this model" if x is None else x
    fields = no_null(fields)
    fp = f"{R}/raw/meta_fields.json"
    json.dump(fields, open(fp, "w"), indent=1, default=str)
    scan = f"{R}/raw/secrets_scan.csv"
    files = [p for p in glob.glob(f"{R}/**", recursive=True) if os.path.isfile(p)]
    subprocess.run([a.python, "bench/secrets_scan.py", "--out", scan, "--token-file", os.path.expanduser("~/.cache/huggingface/token")]
                   + files, check=False)
    subprocess.run([a.python, "bench/meta.py", "--run", R, "--fields", fp, "--device-state", "--scan", scan], check=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
