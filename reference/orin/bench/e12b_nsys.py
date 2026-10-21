"""E12b M2 step 5 -- one Nsight Systems capture of the M2 configuration.

Run UNDER nsys with --capture-range=cudaProfilerApi --cuda-graph-trace=node. Captures, each in
its own NVTX range after full warm-up: `pipeline` (5 frames), `prefix_alone` (5), `expert_alone`
(5), `serial` (3). Timings for every ratio come from the unprofiled runs, never from this
capture (nsys inflates this launch-heavy workload ~1.5x, E11f); this capture explains.

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node -t cuda,nvtx -o scratch/e12b/nsys_m2 \
        python bench/e12b_nsys.py [--config graphs_both --split 12/4]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rung_prefix", choices=("rung_prefix", "graphs_both"))
    ap.add_argument("--split", default="")
    args = ap.parse_args()
    policy, spec, cfg, note = C.load_rung(10)
    H.recompile_default(policy, spec, "denoise_step")
    if args.config == "graphs_both":
        H.recompile_default(policy, spec, "embed_image")
        H.memo_681(policy)
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    sp = tuple(int(x) for x in args.split.split("/")) if args.split else None
    h = H.Harness(policy, spec, frames, 10, config=args.config, split=sp, order="prefix_first")
    fin = C.Finite()
    with torch.no_grad():
        for c in ("pipeline", "prefix_alone", "expert_alone", "serial"):
            h.run(c, 6, fin)                      # warm every condition before the capture
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for c, F in (("pipeline", 5), ("prefix_alone", 5), ("expert_alone", 5), ("serial", 3)):
            torch.cuda.nvtx.range_push(c)
            h.run(c, F, fin)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    print(f"  captured {h.label()}; nonfinite {fin.read()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
