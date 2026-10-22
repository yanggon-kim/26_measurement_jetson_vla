"""T001 step 4c-4f: toy ncu targets.

--case linear   one bf16 F.linear, x [51, 1024], W [4096, 1024], inside NVTX "toy" between cudaProfilerStart/Stop
--case tiny     1,000 tiny elementwise kernels inside NVTX "toy" (ncu per-process kernel limit, step 4f)
--case graph    the linear captured in a torch.cuda.CUDAGraph, replayed 3x inside NVTX "toy" (graph-node visibility)

ncu --set full --nvtx --nvtx-include "toy/" --profile-from-start off -o results/<run>/raw/ncu_toy -f \
    python bench/ncu_toy.py --case linear
"""
import argparse

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="linear", choices=["linear", "tiny", "graph"])
    args = ap.parse_args()
    torch.manual_seed(0)
    x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    t = torch.zeros(256, device="cuda")
    for _ in range(20):
        F.linear(x, w)
        t.add_(1.0)
    torch.cuda.synchronize()
    if args.case == "graph":
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                F.linear(x, w)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = F.linear(x, w)
        torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("toy"):
        if args.case == "linear":
            y = F.linear(x, w)
        elif args.case == "tiny":
            for _ in range(1000):
                t.add_(1.0)
            y = t
        else:
            for _ in range(3):
                g.replay()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("ok", args.case, tuple(y.shape), bool(torch.isfinite(y).all()))


if __name__ == "__main__":
    main()
