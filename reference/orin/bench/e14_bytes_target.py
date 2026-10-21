"""T016 step 3 (A4) / T017 2b: DRAM bytes per frame of pi0's prefix and expert per padding VARIANT,
by the E12b method (bench/run_ncu_e12b_dram.sh): the E12b graphs_both harness -- the whole prefix ONE
CUDA graph (owngraph_prefix + memo_681) and the expert ONE graph (owngraph_expert), whole GPU -- run
under `ncu --graph-profiling graph --cache-control none` with the E12b lts__ metric set, so every
stage frame is one profiled workload (far below the 480-kernel-per-process limit that stops per-kernel
captures on this device).

One Harness per variant (frames trimmed by bench/pi0_padtrim.py BEFORE the graphs are captured, so
each variant's graphs are built at its own static shapes). NVTX ranges <variant>_prefix (5 frames of
prefix_alone) and <variant>_expert (expert_alone: the harness first fills both KV buffers with two
prefix frames INSIDE the range -- the analyzer drops those two graph workloads, as for E12b).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import pi0_padtrim as PT         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pad,nopad")
    ap.add_argument("--frames", type=int, default=5)
    a = ap.parse_args()
    import torch._dynamo as dyn                                                   # noqa: PLC0415
    dyn.config.automatic_dynamic_shapes = False
    dyn.config.cache_size_limit = max(32, dyn.config.cache_size_limit)
    policy, spec, cfg, note = C.load_rung(10)
    H.recompile_default(policy, spec, "denoise_step")
    H.recompile_default(policy, spec, "embed_image")
    H.memo_681(policy)
    base, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    fin = C.Finite()
    hs = {}
    with torch.no_grad():
        for v in a.variants.split(","):
            fr = [{**f, "ctx": PT.ctx_for(f["ctx"], v)} for f in base]
            hs[v] = H.Harness(policy, spec, fr, 10, config="graphs_both", order="prefix_first")
            for c in ("prefix_alone", "expert_alone", "serial"):
                hs[v].run(c, 6, fin)
            print(f"  {v}: harness built ({hs[v].label()})", flush=True)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for v, h in hs.items():
            for c in ("prefix", "expert"):
                torch.cuda.nvtx.range_push(f"{v}_{c}")
                h.run(f"{c}_alone", a.frames, fin)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    print(f"  captured {list(hs)}; nonfinite {fin.read()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
