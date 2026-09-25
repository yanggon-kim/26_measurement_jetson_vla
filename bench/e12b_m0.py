"""E12b M0 -- is a CUDA-graph REPLAY confined to its green-context partition?

T014 verified green-context confinement with EAGER launches (a 2048^2 bf16 GEMM 3.245x / 1.637x
slower on 4 / 8 SMs). E12b's co-runs replay CUDA graphs, and a graph captured on one context and
launched on another may or may not be confined (spec §6 M0). So the same GEMM is captured as a
graph of K back-to-back GEMMs and timed along every launch path the co-run could use:

  eager_green          eager launches on the green stream, green context current (T014's path)
  graph_primary_on_green   captured on the PRIMARY context, replayed on the green stream with the
                       green context current
  graph_green          captured on the green stream with the green context current, replayed there
  graph_green_primary_current  captured on green, replayed with the PRIMARY context current

A path is CONFINED if its time scales like the eager-green reference (roughly 16/n). A path that
runs at full-GPU speed from a small partition is NOT confined, whatever the stream says, and no
co-run number may be taken through it. Also records the split sizes the driver grants.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import green_ctx as G      # noqa: E402

K = 20     # GEMMs per graph


def timed(fn, stream, reps=15):
    ts = []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(stream)
        fn()
        e1.record(stream)
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / K)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 2048
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    primary = torch.cuda.current_stream()

    def body():
        for _ in range(K):
            torch.mm(a, b, out=c)

    rows = []
    # full GPU reference: eager and a graph on the primary context
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        body(); torch.cuda.synchronize()
        g_full = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_full, stream=side):
            body()
    ref_eager = timed(body, primary)
    ref_graph = timed(g_full.replay, primary)
    rows.append({"path": "eager_primary", "sms": 16, "ms_per_gemm": ref_eager})
    rows.append({"path": "graph_primary", "sms": 16, "ms_per_gemm": ref_graph})
    print(f"  16 SMs: eager {ref_eager:.4f}  graph {ref_graph:.4f} ms/GEMM")

    keep = []
    for sms in (4, 8, 12):
        gs = G.GreenCtxStream(sms)
        got = gs.granted_sm
        # eager on green (T014's verified path)
        with gs.activate() as st:
            t = timed(body, st)
        rows.append({"path": "eager_green", "sms": got, "ms_per_gemm": t})
        # graph captured on primary, replayed on green with green current
        try:
            with gs.activate() as st:
                t = timed(lambda: g_full.replay(), st)
            # NB: CUDAGraph.replay() launches on the CURRENT torch stream = st here
            rows.append({"path": "graph_primary_on_green", "sms": got, "ms_per_gemm": t})
        except Exception as e:            # noqa: BLE001
            rows.append({"path": "graph_primary_on_green", "sms": got, "error": str(e)[:160]})
        # graph captured on green with green current, replayed there
        try:
            with gs.activate() as st:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=st):
                    body()
                t = timed(g.replay, st)
            rows.append({"path": "graph_green", "sms": got, "ms_per_gemm": t})
            # same graph, replayed on the green stream but PRIMARY context current
            try:
                with torch.cuda.stream(gs.stream):
                    t2 = timed(g.replay, gs.stream)
                rows.append({"path": "graph_green_primary_current", "sms": got,
                             "ms_per_gemm": t2})
            except Exception as e:        # noqa: BLE001
                rows.append({"path": "graph_green_primary_current", "sms": got,
                             "error": str(e)[:160]})
        except Exception as e:            # noqa: BLE001
            rows.append({"path": "graph_green", "sms": got, "error": str(e)[:160]})
        # A graph captured INSIDE a green context must be freed BEFORE that context is
        # destroyed: the first version called gs.close() with it alive and segfaulted on the
        # next cuGreenCtxDestroy / graph teardown. Same rule for the co-run harness.
        g = None
        torch.cuda.synchronize()
        import gc; gc.collect()
        keep.append(gs)          # contexts are kept alive to process exit, never destroyed early
        for r in rows[-4:]:
            if "ms_per_gemm" in r:
                print(f"  {r['sms']:2d} SMs {r['path']:28s} {r['ms_per_gemm']:.4f} ms/GEMM "
                      f"slowdown {r['ms_per_gemm'] / ref_eager:.3f}x (ideal {16 / r['sms']:.2f}x)")
            else:
                print(f"  {r['sms']:2d} SMs {r['path']:28s} ERROR {r['error']}")

    for r in rows:
        if "ms_per_gemm" in r:
            r["slowdown_vs_full"] = round(r["ms_per_gemm"] / ref_eager, 4)
            r["ideal"] = round(16 / r["sms"], 3)
            # confined if it slows at least half as much as ideal, beyond 1.0
            r["confined"] = bool(r["sms"] == 16 or r["slowdown_vs_full"] - 1 >= 0.5 * (r["ideal"] - 1))
    splits = {}
    for a_, b_ in ((12, 4), (8, 8), (10, 6), (14, 2), (4, 12), (6, 10), (2, 14)):
        try:
            x, y = G.split_pair(a_, b_)
            splits[f"{a_}/{b_}"] = f"{x.granted_sm}/{y.granted_sm}"
            keep.extend([x, y])
        except Exception as e:            # noqa: BLE001
            splits[f"{a_}/{b_}"] = f"FAILED: {e}"
    out = {"gemm": f"{n}x{n} bf16, {K} per graph", "rows": rows, "granted_splits": splits,
           "single_partitions_granted": "(4,4,4,4) (6,6) (8,8) (10,) (12,) per T014 probe",
           "torch": torch.__version__}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("  splits granted:", splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
