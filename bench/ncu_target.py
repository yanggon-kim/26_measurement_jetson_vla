"""T004 / T007 — the workload Nsight Compute profiles, one region at a time.

Spec: `docs/03_instruction_level_analysis.md` §11 item 1 — args `--model --mode --num-steps
--warmup --region`, and it must run standalone.

Protocol §3.2 forbids an unfiltered `--set full`: every ncu command carries one NVTX range or
one kernel regex plus `-c`/`-s` limits. This target warms up, then wraps exactly ONE inference
in `cudaProfilerStart/Stop` with `bench/regions.py`'s NVTX ranges on it, so ncu sees one
instance of the requested region.

The region names available are the ones `bench/regions.py` emits (each also pushed without
brackets, because §3.2 warns `--nvtx-include` may not parse them):

    vision            vlm_prefix            ae_step[i] / ae_step_i     ae_layer[j] / ae_layer_j

`ae_total` is NOT an NVTX range — it is derived from the first and last `ae_step` events — so
`--nvtx-include "ae_total/"` would match nothing.

Typical use, smallest region first (protocol §3.2's escalation order):

    sudo $(which ncu) --profile-from-start off --nvtx --nvtx-include "ae_layer[0]/" \\
        -c 1 --set full --csv --page details --log-file <out>/ncu/ae_layer0_details.csv \\
        --export <out>/ncu/ae_layer0 --force-overwrite \\
        python bench/ncu_target.py --region "ae_layer[0]" --mode eager --num-steps 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="model key from bench/model_registry.py")
    ap.add_argument("--mode", default="eager",
                    help="eager, or a torch.compile mode such as reduce-overhead")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--region", default="ae_layer[0]",
                    help="informational: the NVTX range ncu is filtering on. The target emits "
                         "all of them; ncu selects.")
    ap.add_argument("--no-regions", action="store_true",
                    help="do not instrument. Required for compiled captures: the context "
                         "managers graph-break the model (bench/run_e4.py).")
    args = ap.parse_args()

    gate = P.gate()
    cm = None if args.mode == "eager" else args.mode
    if cm:
        # SmolVLA's `best` captures silently measured EAGER: dynamo hit
        #   torch._dynamo hit config.recompile_limit (8)  function: 'sample_actions'
        # and fell back, so the compiled and eager captures had byte-identical kernels and
        # counters. A capture that quietly measures the wrong configuration is worse than
        # one that fails, so the limit is raised and the outcome is recorded below.
        import torch._dynamo as _dyn
        # 16, not 64. At 8 dynamo BAILED to eager and the capture silently measured the
        # wrong configuration. At 64 it went the other way: SmolVLA's sample_actions
        # recompiled for 27 minutes after profiling had already finished, blocking the
        # queue. 16 clears the bail threshold while bounding the thrash.
        for attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(_dyn.config, attr):
                setattr(_dyn.config, attr, 16)
    spec = MR.get(args.model)
    policy, _, _ = MR.load_policy(spec, n=args.num_steps, compile_mode=cm)
    # assert_real_weights diffs sampled tensors against PI0's OWN safetensors, so it is
    # pi0-specific: on pi0.5 it reported "0/357 tensors match" and aborted the capture,
    # which is a false alarm about a real guard rather than a real failure. It exists
    # because PI0Policy.from_pretrained once swallowed a load error and returned RANDOM
    # weights while still producing plausible latencies (env/README.md §10), so it is kept
    # for the model it can actually check and recorded as unverified elsewhere.
    # SmolVLA cannot stream-capture its vision tower: SmolVLMVisionEmbeddings.forward
    # ends in a boolean-mask index_put whose output shape depends on the mask's VALUES, so
    # it needs a device-to-host sync and any CUDA-graph capture around it dies with
    # "operation not permitted when stream is capturing". All three smolvla `best` C1
    # captures failed that way. Part B solved it with the vis_static_pos rung: in LeRobot's
    # SmolVLA the mask is all-true BY CONSTRUCTION (embed_image passes None;
    # modeling_smolvlm builds torch.ones), so the statement is exactly
    # position_ids = pos_ids. The same prep is applied here, and only when graphs are
    # actually in play, so the eager captures are untouched.
    if args.model == "smolvla" and cm and "reduce-overhead" in cm:
        import opt_ladder as OL  # noqa: E402
        print("prep:", OL.prep_smolvlm_static_posids(policy), flush=True)

    weights = (P.assert_real_weights(policy) if args.model == "pi0" else
               {"verdict": "not verified: assert_real_weights diffs against pi0's own "
                           "safetensors and is pi0-specific"})
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)

    rec = undo = None
    if not args.no_regions:
        need_layers = args.region.startswith("ae_layer")
        # `layers_in_step` is the STEP INDEX whose layers get ranges, not a layer count
        # (regions.py:138 compares it with rec.step_idx). Passing 18 asked for the layers of
        # step 18, which does not exist at N=4, so no ae_layer range was ever emitted and ncu
        # reported "No kernels were profiled".
        rec = R.Recorder(enable_layers=need_layers, layers_in_step=0)
        undo = R.instrument(policy, rec, spec.container)

    for _ in range(args.warmup):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
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

    print(json.dumps({
        "region_requested": args.region, "mode": args.mode, "num_steps": args.num_steps,
        "weights": weights,
        # Positive evidence that compilation actually happened, so a fallback to eager
        # cannot pass as a "best" capture again.
        "dynamo": (lambda c: {"frames_compiled": int(sum(c.get("frames", {}).values())),
                              "graph_breaks": int(sum(c.get("graph_break", {}).values())),
                              "compiled": bool(c.get("frames"))})(
            __import__("torch")._dynamo.utils.counters) if cm else {"compiled": False},
        "warmup": args.warmup, "instrumented": rec is not None,
        "gpu_cur_freq_hz": gate["gpu_cur_freq_hz"], "power_mode": gate["power_mode"],
        "sm_count": gate["sm_count"],
        "regions_ms": None if rec is None else rec.results_ms(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
