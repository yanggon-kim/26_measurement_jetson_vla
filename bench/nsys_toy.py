"""T001 step 4b/6: toy nsys target. Warm-up outside the capture window, then 10 bf16 linears
(x [51, 1024], W [4096, 1024]) inside NVTX range "toy" between cudaProfilerStart/Stop.

nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
     -o scratch/<run>/nsys_toy python bench/nsys_toy.py
"""
import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(0)
    x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        F.linear(x, w)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("toy"):
        for _ in range(10):
            y = F.linear(x, w)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("ok", tuple(y.shape), bool(torch.isfinite(y).all()))


if __name__ == "__main__":
    main()
