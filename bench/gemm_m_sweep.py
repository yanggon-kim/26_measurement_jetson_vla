"""T005 diagnostic: time one bf16 GEMM shape over a range of M (token counts) in isolation, and name the kernel cuBLAS
picks for each M. Shape: x [M, K] @ W^T with W [N, K] (torch.nn.functional.linear, as the model calls it).
CUDA events, warm-up 10, timed 50 per M (median); one torch.profiler call per M for the kernel name.

python bench/gemm_m_sweep.py --k 2048 --n 16384 --m 512:1024:8 --out results/<dir>/gemm_m_sweep_gateup.csv
"""
import argparse
import csv
import statistics
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--m", default="512:1024:8", help="start:stop:step (inclusive) or a comma list")
    ap.add_argument("--blas", default="default", choices=["default", "cublas", "cublaslt"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import gate as G
    G.wait_for_gate(apply_clocks=True)
    if a.blas != "default":
        torch.backends.cuda.preferred_blas_library(a.blas)
    ms = ([int(x) for x in a.m.split(",")] if "," in a.m else
          list(range(int(a.m.split(":")[0]), int(a.m.split(":")[1]) + 1, int(a.m.split(":")[2]))))
    w = torch.randn(a.n, a.k, device="cuda", dtype=torch.bfloat16)
    rows = []
    from torch.profiler import ProfilerActivity, profile
    for m in ms:
        x = torch.randn(m, a.k, device="cuda", dtype=torch.bfloat16)
        f = lambda: torch.nn.functional.linear(x, w)  # noqa: E731
        for _ in range(10):
            f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            f()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            f()
            torch.cuda.synchronize()
        kn = ";".join(ev.name for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA)
        med = statistics.median(ts)
        rows.append({"M": m, "K": a.k, "N": a.n, "blas": a.blas, "median_ms": round(med, 4), "min_ms": round(min(ts), 4),
                     "TFLOPps": round(2 * m * a.k * a.n / med / 1e9, 2), "kernels": kn})
        print(f"M={m:5d} {med:7.3f} ms {2 * m * a.k * a.n / med / 1e9:6.1f} TFLOP/s  {kn[:90]}", flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
