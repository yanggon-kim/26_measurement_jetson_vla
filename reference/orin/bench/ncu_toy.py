"""T001 step 3b/3c — the toy profiler target.

Warm-up (protocol §2: >= 20 iterations) happens OUTSIDE the capture window; then
between cudaProfilerStart/Stop it runs bf16 `F.linear` with x [S, K], W [N, K]
inside the NVTX range `toy` (and the alias `toy_alias` without brackets, so that the
bracket handling of --nvtx-include can be checked on a range that has none).

Defaults are the shapes T001 asks for: x [51, 1024], W [4096, 1024] — the mlp_gate /
mlp_up shape of the pi0 action expert at its own suffix length.

Used by:
  nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi -f true \
       -o <out> python bench/ncu_toy.py --iters 3
  ncu --set full --nvtx --nvtx-include "toy/" --profile-from-start off \
       -o <out> -f python bench/ncu_toy.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nvtx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=51, help="rows of x (suffix tokens)")
    ap.add_argument("--K", type=int, default=1024, help="input features")
    ap.add_argument("--N", type=int, default=4096, help="output features")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=1,
                    help="calls inside the capture window (ncu: 1, nsys: 1-3)")
    ap.add_argument("--no-profiler-api", action="store_true",
                    help="run without cudaProfilerStart/Stop (plain functional check)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(args.S, args.K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(args.N, args.K, device=dev, dtype=torch.bfloat16)

    for _ in range(args.warmup):
        y = F.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.time()

    if not args.no_profiler_api:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        with nvtx("toy"):
            y = F.linear(x, w)
        with nvtx("toy_alias"):
            y2 = F.linear(x, w)
    if not args.no_profiler_api:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()

    print(f"ok shape={tuple(y.shape)} dtype={y.dtype} "
          f"finite={bool(torch.isfinite(y).all().item())} "
          f"capture_window_s={time.time() - t0:.3f} "
          f"x={tuple(x.shape)} W={tuple(w.shape)} iters={args.iters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
