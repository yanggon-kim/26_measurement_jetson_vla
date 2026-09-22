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

    for _ in range(3):
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

    rows = [{"kernel": n, "launches": c, "launches_per_step": round(c / args.steps, 2)}
            for n, c in names.most_common()]
    with (out / "kernels_per_step.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "launches", "launches_per_step"])
        w.writeheader()
        w.writerows(rows)

    total = sum(names.values())
    print(f"  distinct kernel names : {len(names)}")
    print(f"  total kernel launches : {total} for one inference with N={args.steps} steps")
    print(f"  per denoising step    : {total/args.steps:.0f} (upper bound: the prefix and "
          f"vision run once per inference, not per step)")
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
        "compile_mode": None, "batch": 1, "num_inference_steps": args.steps,
        "n_warmup": 3, "n_timed": 1,
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
        "launches_per_step_upper_bound": round(total / args.steps, 1),
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
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
