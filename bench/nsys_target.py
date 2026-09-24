"""T010 / T004 — the thing `bench/run_nsys.sh` profiles: N steady-state pi0 inferences.

Warms up, then wraps exactly `--inferences` calls in `cudaProfilerStart/Stop` so the nsys
capture range holds only steady-state work, with `bench/regions.py`'s NVTX ranges on it.

    python bench/nsys_target.py --model pi0 --steps 10 --inferences 2
    python bench/nsys_target.py --model pi0 --steps 10 --rung bf16_both
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
    ap.add_argument("--model", default="pi0", help="pi0 | pi05 | smolvla")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--inferences", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--compile-mode", default=None,
                    help="applied to sample_actions. This SWALLOWS the whole frame, so "
                         "the per-region NVTX ranges cannot survive it -- use --rung "
                         "instead for a compiled capture that keeps its regions.")
    ap.add_argument("--rung", default=None,
                    help="an opt_ladder rung (e.g. bf16_both, vlm_graphs). The rungs "
                         "compile embed_image / container.forward / denoise_step, and "
                         "the regions are instrumented AFTER compiling, so every region "
                         "boundary stays OUTSIDE a compiled callable and the per-region "
                         "split SURVIVES. This is how Part B has per-region numbers for "
                         "its compiled rungs, and it is what makes E5's compiled half "
                         "measurable at all.")
    ap.add_argument("--no-regions", action="store_true",
                    help="regions.py's context managers graph-break a model compiled "
                         "through --compile-mode; needed only for that path")
    args = ap.parse_args()
    gate = P.gate()
    import model_registry as MR                                       # noqa: PLC0415
    spec = MR.get(args.model)
    policy, _, _ = MR.load_policy(spec, n=args.steps,
                                  compile_mode=None if args.rung else args.compile_mode)
    P.assert_real_weights(policy, spec=spec)   # this model's own checkpoint
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)

    rung_desc = ""
    if args.rung:
        import opt_ladder as OL                                       # noqa: PLC0415
        compiled, prep_note, prec, comp_s, fails, regions_ok = OL.apply_rung(
            policy, spec, args.rung)
        if not regions_ok and not args.no_regions:
            # This rung compiles sample_actions as one unit, so a region context manager
            # inside it graph-breaks the model. Instrumenting anyway produces a capture
            # with no per-region split and nothing saying why -- which is the state
            # 20260922_E5_pi0_reduce-overhead_N4 has been in since it was abandoned.
            print(f"  rung {args.rung} has regions_ok=False (it compiles sample_actions "
                  f"as one unit); running WITHOUT regions. The capture will have a "
                  f"whole_capture row and no per-region split.", flush=True)
            args.no_regions = True
        if fails:
            raise SystemExit(f"rung {args.rung!r} failed to compile: {fails}. "
                             f"A partially-applied rung is not the rung.")
        rung_desc = f"{args.rung}: prep=[{prep_note}] compiled=[{';'.join(compiled)}]"
        print(rung_desc, flush=True)

    # Instrument AFTER compiling, always. That ordering is the whole trick: it keeps every
    # region boundary outside a compiled callable, so the ranges are not graph-broken away.
    rec = undo = None
    if not args.no_regions:
        rec = R.Recorder()
        undo = R.instrument(policy, rec, spec.container)
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
    print(json.dumps({"gate": gate, "model": args.model, "steps": args.steps,
                      "inferences": args.inferences,
                      "compile_mode": None if args.rung else args.compile_mode,
                      "rung": args.rung, "rung_detail": rung_desc,
                      "regions": None if rec is None else rec.results_ms()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
