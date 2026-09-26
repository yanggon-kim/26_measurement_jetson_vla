"""T016 step 1d (EV-02c): the pi0 bf16_both rung PREFIX, run for ncu (per kernel) or torch.profiler.

--mode ncu      warm up, then cudaProfilerStart and `--frames` prefix frames, each in an NVTX range
                `prefix_f<k>`; run under `ncu --profile-from-start off --nvtx --graph-profiling node`.
--mode shapes   one prefix frame under torch.profiler(record_shapes=True): every CPU op that launched a
                kernel, in launch order, with its input shapes and the kernel names -> CSV. This is the
                op / GEMM-shape map for the ncu rows (matched by launch order and kernel name). Kernels
                replayed inside a CUDA graph (embed_image@reduce-overhead) are invisible to the profiler
                and are mapped from --mode shapes with --vision-nograph (embed_image compiled
                mode="default", same Inductor kernels, no graph), recorded as such.
--vision-nograph   compile embed_image mode="default" instead of reduce-overhead (fallback if ncu node
                profiling of the graph fails; the kernel choice may differ -- totals are compared).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ncu", "shapes"), required=True)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--vision-nograph", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    policy, spec, cfg, note = C.load_rung(10)
    if args.vision_nograph:
        note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    with torch.no_grad():
        for k in range(20):
            CR.prefix_stage(frames[k % 5]["ctx"])
        torch.cuda.synchronize()
        if args.mode == "ncu":
            torch.cuda.cudart().cudaProfilerStart()
            for k in range(args.frames):
                torch.cuda.nvtx.range_push(f"prefix_f{k}")
                CR.prefix_stage(frames[k % 5]["ctx"])
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("  captured", args.frames, "prefix frames;", note, flush=True)
            return 0
        from torch.profiler import ProfilerActivity, profile               # noqa: PLC0415
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            CR.prefix_stage(frames[0]["ctx"])
            torch.cuda.synchronize()
    rows = []
    evs = sorted((e for e in prof.events() if e.device_type.name == "CPU" and e.kernels),
                 key=lambda e: e.time_range.start)
    for e in evs:
        for kr in e.kernels:
            rows.append({"cpu_op": e.name, "input_shapes": str(e.input_shapes),
                         "kernel": kr.name, "cpu_start_us": round(e.time_range.start, 3)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} kernel launches with ops/shapes -> {out}; {note}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
