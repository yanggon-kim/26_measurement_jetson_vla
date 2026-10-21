"""E12d arm (a) -- one Nsight Systems capture of T014's EAGER in-process co-run, to measure overlap directly.

T014 concluded "no overlap because the host issues for the whole wall clock" from issue/wall = 1.0000,
which the review rejected as proof (it also counts launch-queue back-pressure). This capture measures
the thing itself: the fraction of each co-run frame during which kernels of BOTH streams execute, and
each stream's in-stage gaps, with bench/e12b_nsys_analyze.py (the E12b M2 step 5 analysis, unchanged).

Configuration = T014's committed eager run (results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync):
pi0 eager, N = 10, corun.patch_pi0_nosync (label pi0_nosync), corun.prefix_stage / expert_stage, the
pipeline exactly corun.run_pipeline (expert(k-1) issued before prefix(k), gated by events).

NVTX ranges, each after warm-up, in the names and frame counts the analyser expects:
  pipeline      corun.run_pipeline over 5 frames      (prefix stream + expert stream)
  prefix_alone  5 prefixes on the prefix stream
  expert_alone  5 experts on the expert stream, caches computed BEFORE the range
  serial        3 frames, prefix then expert, synchronised per frame

Timings for ratios never come from this capture (nsys inflates launch-heavy work); it explains.

    bash bench/run_nsys.sh <out> e12d_eager -- python bench/e12d_nsys.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import pi0_common as P           # noqa: E402

N = 10


def main() -> int:
    import model_registry as MR                                        # noqa: PLC0415
    g = P.gate()
    print("gate:", g, flush=True)
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=N, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    print("  " + CR.patch_pi0_nosync(policy), flush=True)
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    sp, se = torch.cuda.Stream(), torch.cuda.Stream()
    nonfinite = 0
    with torch.no_grad():
        # warm every path
        CR.run_serial(frames[:3], N)
        CR.run_pipeline(frames, N, sp, se)
        torch.cuda.synchronize()
        caches = []
        for fr in frames:
            with torch.cuda.stream(sp):
                caches.append(CR.prefix_stage(fr["ctx"]))
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("pipeline")
        acts, _ = CR.run_pipeline(frames, N, sp, se, keep_actions=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("prefix_alone")
        with torch.cuda.stream(sp):
            for fr in frames:
                CR.prefix_stage(fr["ctx"])
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("expert_alone")
        with torch.cuda.stream(se):
            for fr, (pkv, pad) in zip(frames, caches):
                a = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N)
                nonfinite += int((~torch.isfinite(a)).sum().item())
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("serial")
        for fr in frames[:3]:
            with torch.cuda.stream(sp):
                pkv, pad = CR.prefix_stage(fr["ctx"])
            se.wait_stream(sp)
            with torch.cuda.stream(se):
                a = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N)
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        nonfinite += sum(int((~torch.isfinite(v)).sum().item()) for v in acts.values())
    print(f"  captured pi0 eager pi0_nosync co-run; non-finite action elements {nonfinite}", flush=True)
    return 0 if nonfinite == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
