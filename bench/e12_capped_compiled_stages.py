"""T014 Step 4 -- the COMPILED stages alone, at any power mode, for the capped-mode cross-check.

analysis/E12b_review.md: the restated Step-4 prediction quoted compiled 8-SM references, but
bench/queues/queue_e12_capped.sh never measured compiled stages at MODE_30W, so those rows could be
neither tested nor falsified. This script measures them, through the SAME E12b harness and
configuration as the confined MAXN references (graphs_both: owngraph_prefix + memo_681 +
owngraph_expert + dbuf_kv, bf16_both rung, hoist, N = 10):

    prefix_alone, expert_alone and serial, --frames frames each, CUDA events on each stage's
    stream, device-side non-finite count on every action tensor.

At MAXN it can also run inside a green split (--split 8/8) -- the confined reference the review
names (A_part 326.94, B_part 149.05 ms in results/20260925_E12b_pi0_green_corun_N10_MAXN). At
MODE_30W the whole GPU IS 8 SMs in hardware, so --split is refused there.

Prediction (STATUS.md, restated after the review): a memory-bound stage lies BETWEEN no change and
the GPU-clock ratio; the EMC rate is recorded so the bracket can be read.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import pi0_common as P           # noqa: E402

N = 10


def state() -> dict:
    g = P.gate(require_maxn=False)
    try:
        g["emc_rate_hz"] = int(subprocess.run(
            ["sudo", "-n", "/bin/sh", "-c", "cat /sys/kernel/debug/bpmp/debug/clk/emc/rate"],
            capture_output=True, text=True, timeout=10).stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        g["emc_rate_hz"] = "unreadable"
    g["cpu_cur_freq_khz_each"] = {
        p.parent.parent.name: p.read_text().strip()
        for p in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"))}
    g["temps_mC"] = C.temps()
    g["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--split", default="", help="MAXN only, e.g. 8/8")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    g0 = state()
    maxn = "MAXN" in g0["power_mode"]
    if args.split and not maxn:
        raise SystemExit("--split is a MAXN reference; at a capped mode the whole GPU is the gated one")

    policy, spec, cfg, note = C.load_rung(N)
    note["expert"] = H.recompile_default(policy, spec, "denoise_step")
    note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
    note["memo"] = H.memo_681(policy)
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    sp = tuple(int(x) for x in args.split.split("/")) if args.split else None
    h = H.Harness(policy, spec, frames, N, config="graphs_both", split=sp, order="prefix_first")
    fin = C.Finite()
    rows, samples = [], []
    with torch.no_grad():
        for c in ("prefix_alone", "expert_alone", "serial"):
            h.run(c, 6, fin)                                  # warm-up / capture, untimed
        torch.cuda.synchronize()
        for c, key in (("prefix_alone", "prefix_ms"), ("expert_alone", "expert_ms"),
                       ("serial", "frame_ms")):
            r = h.run(c, args.frames, fin)
            v = [x[key] for x in r["records"]]
            s = C.summ(v)
            rows.append({"condition": c, "stage": key.replace("_ms", ""), **s,
                         "period_ms": round(r["period_ms"], 4),
                         "host_issue_median_ms": round(sorted(r["host_issue_ms"])[len(v) // 2], 4)})
            samples += [{"condition": c, "frame": i, "ms": round(x, 5)} for i, x in enumerate(v)]
            print(f"  {c:13s} median {s['median_ms']:9.3f} ms  (p5 {s['p5_ms']:.3f}, "
                  f"p95 {s['p95_ms']:.3f}, n {s['n']})", flush=True)
    nonfinite = fin.read()
    g1 = state()
    with (out / "stages.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "frame", "ms"])
        w.writeheader()
        w.writerows(samples)
    throttled = g1["gpu_cur_freq_hz"] != g0["gpu_cur_freq_hz"]
    C.write_meta(out / "meta.json", {
        "experiment": "T014 Step 4 -- compiled stages alone (E12b harness, graphs_both)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
        "rung": "bf16_both", "rung_note": note, "variant": h.label(),
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": f"{h.granted[0]}/{h.granted[1]}" if sp else "none (whole GPU)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"],
        "throttled": bool(throttled),
        "throttled_evidence": f"GPU clock start {g0['gpu_cur_freq_hz']} end {g1['gpu_cur_freq_hz']}",
        "gate_start": g0, "gate_end": g1, "N": N, "frames": args.frames,
        "n_nonfinite": nonfinite, "valid": nonfinite == 0,
        "maxn_references": {
            "whole_gpu_graphs_both": {"A_iso_ms": 246.2792, "B_iso_ms": 101.3649,
                                      "source": "results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv (full_gpu)"},
            "green_8_8": {"A_part_ms": 326.9407, "B_part_ms": 149.05,
                          "source": "same file, split 8/8"}},
        "command": " ".join([sys.executable, *sys.argv]), "torch": torch.__version__,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"  nonfinite {nonfinite}; wrote {out}", flush=True)
    return 0 if nonfinite == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
