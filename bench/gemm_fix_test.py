"""T005 diagnostic: can the slow cuBLASLt kernel choice for pi0's MLP gate/up GEMM (M = 816, K = 2048, N = 16384, bf16)
be avoided? Each remedy is timed on the same operands (CUDA events, warm-up 10, timed 50, median) and its output is
compared with the default F.linear (max |diff|, relative L2):
  default        F.linear (cuBLASLt heuristic)
  tunableop      PyTorch TunableOp: times cuBLAS / cuBLASLt candidate algorithms at run time and keeps the fastest
  inductor_at    torch.compile(F.linear, mode="max-autotune") with Inductor's 68-SM gate overridden (ATEN + TRITON)
  pad_<m2>       x zero-padded to m2 rows, F.linear, first M rows kept
  split_<a>      F.linear on x[:a] and x[a:], concatenated
Run each remedy in its own process (--remedy) so TunableOp / Inductor state does not leak.

python bench/gemm_fix_test.py --remedy tunableop --out results/<dir>/gemm_fix_tunableop.json
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def timeit(f, reps=50):
    for _ in range(10):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        f()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


def kernels(f):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f()
        torch.cuda.synchronize()
    return [ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remedy", required=True)
    ap.add_argument("--m", type=int, default=816)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import gate as G
    G.wait_for_gate(apply_clocks=True)
    torch.manual_seed(0)
    x = torch.randn(a.m, a.k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(a.n, a.k, device="cuda", dtype=torch.bfloat16) * 0.02
    ref = torch.nn.functional.linear(x, w).float()
    r, note = a.remedy, ""
    if r == "default":
        f = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
    elif r == "tunableop":
        import torch.cuda.tunable as T
        T.enable(True)
        T.tuning_enable(True)
        T.set_filename(os.path.join(os.path.dirname(os.path.abspath(a.out)), "tunableop_results%d.csv"))
        f = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
        f()                                   # tunes this shape once
        torch.cuda.synchronize()
        T.tuning_enable(False)
        note = f"TunableOp results: {T.get_results()}"
    elif r == "inductor_at":
        import rungs as RG
        note = RG.force_gemm_autotune()
        f = torch.compile(lambda: torch.nn.functional.linear(x, w), mode="max-autotune-no-cudagraphs")
    elif r.startswith("pad_"):
        m2 = int(r[4:])
        xp = torch.zeros(m2, a.k, device="cuda", dtype=torch.bfloat16)
        xp[:a.m] = x

        def f():
            xp[:a.m].copy_(x)
            return torch.nn.functional.linear(xp, w)[:a.m]
        note = f"pad copy + GEMM at M={m2} + slice, all timed"
    elif r.startswith("split_"):
        s = int(r[6:])
        f = lambda: torch.cat([torch.nn.functional.linear(x[:s], w), torch.nn.functional.linear(x[s:], w)])  # noqa: E731
        note = f"two GEMMs (M={s}, M={a.m - s}) + cat, all timed"
    else:
        raise SystemExit(f"unknown remedy {r}")
    out = f().float()
    med, mn = timeit(f)
    d = out - ref
    res = {"remedy": r, "M": a.m, "K": a.k, "N": a.n, "median_ms": med, "min_ms": mn,
           "TFLOPps": 2 * a.m * a.k * a.n / med / 1e9, "max_abs_diff_vs_default": float(d.abs().max()),
           "rel_l2_vs_default": float(d.norm() / ref.norm()), "kernels": kernels(f), "note": note[:2000]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{r:14s} {med:7.3f} ms {res['TFLOPps']:6.1f} TFLOP/s  maxdiff {res['max_abs_diff_vs_default']:.2e} "
          f"rel {res['rel_l2_vs_default']:.2e}  {';'.join(res['kernels'])[:110]}", flush=True)


if __name__ == "__main__":
    main()
