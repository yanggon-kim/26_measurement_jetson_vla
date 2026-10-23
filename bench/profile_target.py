"""Profiling target for nsys and ncu (T006-T008): one model in one configuration, exactly as T005 timed it.

  --rung eager_fp32 | best_bf16 | <any rung of bench/rungs.py>
      best_bf16 resolves through results/*_E4i_summary/best_bf16.json (bench/opt_ladder.resolve_best) and replays the
      recorded linear_mpad table, so the profiled configuration is the timed one.
The model is put in the rung (rungs.apply_rung: precision, preps, compilation). The protocol §2 regions are installed
AFTER compilation (NVTX ranges vision, vlm_prefix, ae_step_<i>, ...; ncu filters on them). Warm-up runs at least
--warmup inferences and continues until no new graph has been compiled for 5 inferences. Then --inferences
inferences run between cudaProfilerStart / Stop, with cudagraph_mark_step_begin before each, as in the timed loop.

nsys: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node ... \
      python bench/profile_target.py --model pi0 --rung best_bf16 --inferences 2
ncu:  ncu --profile-from-start off --nvtx --nvtx-include "vlm_prefix/" ... python bench/profile_target.py ... --inferences 1
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--inferences", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--warmup-max", type=int, default=80)
    ap.add_argument("--info", default="", help="write the resolved configuration here (json)")
    a = ap.parse_args()
    import torch
    from torch._dynamo.utils import counters
    import opt_ladder as OL
    import regions as R
    import rungs as RG
    import run_e3 as E3
    rung, src, table = a.rung, "", None
    if a.rung == "best_bf16":
        rung, src = OL.resolve_best(a.model)
        table = OL.mpad_table_of(a.model, src)
    ad = E3.make_adapter(a.model)
    ad.name, ad.family = a.model, RG.family_of(a.model)
    if table:
        ad.mpad_fixed = table
    if ad.family == "groot":
        ad.capture_tokens()
    desc = RG.apply_rung(ad, rung)
    ad.set_n(ad.n_default)
    rec = R.Recorder()
    undo = None
    if desc["regions"]:
        if ad.family == "groot":
            undo, _ = R.instrument_groot(ad.policy, rec, inner=not desc["ae_compiled"])
        else:
            undo = R.instrument_lerobot(ad.policy, rec, ad.spec.container)
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    ug = lambda: int(counters["stats"].get("unique_graphs", 0))  # noqa: E731
    hist, it = [ug()], 0
    while it < a.warmup or len(hist) <= 5 or hist[-1] != hist[-6]:
        rec.reset()
        mark()
        ad.iteration()
        hist.append(ug())
        it += 1
        if it >= a.warmup_max:
            break
    torch.cuda.synchronize()
    g0 = ug()
    torch.cuda.profiler.start()
    for i in range(a.inferences):
        rec.reset()
        rec.start()
        mark()
        with torch.cuda.nvtx.range(f"inference_{i}"):
            ad.iteration()
        rec.stop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    info = {"model": a.model, "rung_requested": a.rung, "rung": rung, "best_bf16_source": src,
            "mpad_table_replayed": bool(table), "rung_desc": desc, "warmup_iters": it,
            "graphs_compiled_in_profiled_window": ug() - g0, "region_calls": dict(rec.counts),
            "n_default": ad.n_default, "tokens": {k: v for k, v in ad.tokens.items() if k != "shapes"}}
    if undo:
        undo()
    print(json.dumps({k: info[k] for k in ("rung", "warmup_iters", "graphs_compiled_in_profiled_window", "region_calls")},
                     default=str), flush=True)
    if a.info:
        json.dump(info, open(a.info, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
