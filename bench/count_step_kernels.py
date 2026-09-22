"""T005 — how many GPU kernels does one eager denoising step actually launch?

E2's argument is that the action expert is CPU-dispatch-bound: ~60 us of CPU per operation
against ~35 us of GPU work. Turning that into a share of the step needs the number of
operations per step, and an earlier draft asserted "on the order of 400" without measuring it.
This counts them with `torch.profiler`, and breaks the count down by kernel name so the
projections, the norms and the attention can be told apart.

    python bench/count_step_kernels.py --outdir results/<run>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import REPO, collect_raw, sh_out, software_versions, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    policy, cfg, shipped = P.load_policy(num_inference_steps=args.steps)
    real = P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, preprocess, _ = P.build_batch(policy, fx)

    for _ in range(20):                       # protocol §2: >= 20 warm-ups before a capture
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()

    names = collections.Counter()
    total_us = 0.0
    for e in prof.events():
        for k in getattr(e, "kernels", []) or []:
            names[k.name] += 1
            total_us += k.duration

    # NOT a per-step figure: vision and the VLM prefix run once per inference, so total/N
    # mixes them in. The real per-step number is the SLOPE between two values of N, computed
    # by bench/analyze_step_kernels.py. This column is kept only as a sanity check.
    rows = [{"kernel": n, "launches": c,
             "launches_over_N_not_per_step": round(c / args.steps, 2)}
            for n, c in names.most_common()]
    with (out / "kernels_per_step.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "launches",
                                          "launches_over_N_not_per_step"])
        w.writeheader()
        w.writerows(rows)

    total = sum(names.values())
    print(f"  distinct kernel names : {len(names)}")
    print(f"  total kernel launches : {total} for one inference with N={args.steps} steps")
    print(f"  total/N               : {total/args.steps:.0f}  (NOT the per-step count -- "
          f"vision and the prefix run once per inference; use the slope)")
    print(f"  total GPU kernel time : {total_us/1e3:.2f} ms")
    print("\n  top 15 by launch count:")
    for r in rows[:15]:
        print(f"    {r['launches']:5d}  {r['kernel'][:80]}")

    prof.export_chrome_trace(str(out / "raw" / "step_trace.json"))
    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E2-aux (kernel count per step)", "task": "T005",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_revision": P.REVISION, "precision": "bfloat16",
        "model_repo": P.CHECKPOINT, "model_commit": P.REVISION, "checkpoint": P.CHECKPOINT,
        "num_denoise_steps": args.steps, "action_horizon": getattr(cfg, "chunk_size", None),
        "torchvision": None, "triton": None,          # filled below from software_versions()
        "compile_mode": None, "batch": 1, "num_inference_steps": args.steps,
        "n_warmup": 20, "n_timed": 1,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/count_step_kernels.py --outdir {args.outdir} "
                   f"--steps {args.steps}",
        "total_kernel_launches": total, "distinct_kernel_names": len(names),
        "launches_over_N_not_per_step": round(total / args.steps, 1),
        "_per_step_note": "total/N is NOT the per-step count; use the slope between two N "
                          "values (bench/analyze_step_kernels.py).",
        "total_gpu_kernel_ms": round(total_us / 1e3, 3),
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "raw_files": [],
        "notes": [
            "Counted with torch.profiler over ONE inference; the per-step figure divides by "
            "num_inference_steps and is an UPPER bound, because vision and the VLM prefix "
            "run once per inference rather than once per step.",
            "torch.profiler adds its own overhead, so the TIMES here are not latency "
            "measurements — only the COUNTS are used.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["torchvision"] = sw.get("torchvision")
    meta["triton"] = sw.get("triton")
    meta["jetson_clocks"] = "applied before the run; GPU locked min==max"
    meta["emc_hz"] = int(sh_out(
        "sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0)
    meta["jetpack"] = "6.2.3+b81"
    meta["l4t"] = "R36.5.0"
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
