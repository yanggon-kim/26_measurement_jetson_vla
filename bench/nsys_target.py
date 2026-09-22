"""T010 / T004 — the thing `bench/run_nsys.sh` profiles: N steady-state pi0 inferences.

Warms up, then wraps exactly `--inferences` calls in `cudaProfilerStart/Stop` so the nsys
capture range holds only steady-state work, with `bench/regions.py`'s NVTX ranges on it.

    python bench/nsys_target.py --steps 10 --inferences 2 [--compile-mode reduce-overhead]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--inferences", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--no-regions", action="store_true",
                    help="regions.py's context managers graph-break a compiled model "
                         "(bench/run_e4.py); pass this for compiled captures")
    args = ap.parse_args()
    gate = P.gate()
    policy, _, _ = P.load_policy(num_inference_steps=args.steps,
                                 compile_mode=args.compile_mode)
    P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)

    rec = undo = None
    if not args.no_regions:
        rec = R.Recorder()
        undo = R.instrument(policy, rec)
    for _ in range(args.warmup):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    for _ in range(args.inferences):
        if rec is not None:
            rec.reset()
            rec.start()
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if rec is not None:
            rec.stop()
    cudart.cudaProfilerStop()
    if undo:
        undo()
    print(json.dumps({"gate": gate, "steps": args.steps, "inferences": args.inferences,
                      "compile_mode": args.compile_mode,
                      "regions": None if rec is None else rec.results_ms()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
