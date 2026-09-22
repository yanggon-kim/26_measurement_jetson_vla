"""T005 — is the L2 sysmem-miss counter a usable stand-in for DRAM traffic on this device?

T001 established that this Tegra exposes **no `dram__*` counters at all** (0 of 4426 metrics;
asking for one returns `n/a`). Every off-chip byte the campaign reports therefore has to come
from the L2's sysmem-miss sectors:

    off-chip bytes  ~  lts__t_sectors_aperture_sysmem_*_lookup_miss.sum  x  32 B

The research side asked (protocol §3.3, question 1) for that substitute to be checked against
kernels whose traffic is known by construction, rather than assumed. This script is the
target; `run_ncu_validation.sh` drives ncu over it.

Four cases, each a single kernel run inside NVTX range `micro` with the profiler started only
for that kernel:

  copy_256mib    z = 2*x over 256 MiB of bf16      -> reads == writes == 256 MiB exactly
  gemm_cold      51 x 4096 @ 4096 x 4096           -> streams a 32 MiB weight once, reads >> writes
  gemm_warm      the same kernel, already resident  -> tests whether the counter sees a hit
  gemm_l2        51 x 1024 @ 1024 x 512            -> 1.2 MiB working set, fits in the 4 MiB L2,
                                                      so a warm launch should miss ~nothing

The cold cases must be profiled with ncu's default cache control (it flushes the L2 before
each launch, which is what "cold" means). The warm cases must be profiled with
`--cache-control none`, or ncu's own flush destroys the very thing being measured.

    python bench/ncu_validation.py --case gemm_cold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nvtx  # noqa: E402

MIB = 1024 * 1024


def cases() -> dict:
    """Each entry builds its tensors and returns (run, analytic_read_bytes,
    analytic_write_bytes, description, warm)."""
    def copy_256mib():
        # The task asks for `clone()` of a 256 MiB tensor. On this stack a contiguous
        # device-to-device clone dispatches to cudaMemcpyAsync, which is a copy engine
        # operation and not a kernel -- ncu profiles nothing and reports "No kernels were
        # profiled". An elementwise `z = 2*x` has exactly the same traffic (read 256 MiB,
        # write 256 MiB) and *is* a kernel, so it is used instead. Deviation recorded.
        n = 256 * MIB // 2                       # bf16 elements
        x = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
        z = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        desc = ("z = 2*x over 256 MiB of bf16; reads the source once, writes the output "
                "once (stands in for clone(), which is a memcpy and not a kernel)")
        return (lambda: torch.mul(x, 2.0, out=z)), 256 * MIB, 256 * MIB, desc, False

    def _gemm(S, K, N, warm, desc):
        w = torch.empty(N, K, device="cuda", dtype=torch.bfloat16).normal_()
        x = torch.empty(S, K, device="cuda", dtype=torch.bfloat16).normal_()
        rd = K * N * 2 + S * K * 2
        wr = S * N * 2
        return (lambda: torch.nn.functional.linear(x, w)), rd, wr, desc, warm

    def _gemm2(S, K, N, desc):
        w = torch.empty(N, K, device="cuda", dtype=torch.bfloat16).normal_()
        x = torch.empty(S, K, device="cuda", dtype=torch.bfloat16).normal_()

        def run():
            torch.nn.functional.linear(x, w)
            torch.nn.functional.linear(x, w)

        return run, K * N * 2 + S * K * 2, S * N * 2, desc, True

    return {
        "copy_256mib": copy_256mib,
        "gemm_cold": lambda: _gemm(51, 4096, 4096, False,
                                   "51x4096 @ 4096x4096; streams a 32 MiB weight once"),
        "gemm_warm": lambda: _gemm(51, 4096, 4096, True,
                                   "same kernel as gemm_cold, profiled with the caches left "
                                   "alone (--cache-control none)"),
        # the same small GEMM run TWICE inside the profiled window. If the L2 retained the
        # weight across a kernel boundary, the second launch would miss ~nothing. Profiled
        # with -c 2 so both launches are captured and can be compared.
        "gemm_l2_twice": lambda: _gemm2(51, 1024, 512,
                                        "51x1024 @ 1024x512 run twice back to back; "
                                        "launch 2 tests whether the L2 keeps the weight "
                                        "across a kernel boundary"),
        "gemm_l2": lambda: _gemm(51, 1024, 512, True,
                                 "51x1024 @ 1024x512; 1.2 MiB working set, fits inside the "
                                 "4 MiB L2, so a warm launch should miss almost nothing"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(cases()))
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    run, rd, wr, desc, warm = cases()[args.case]()
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    with nvtx("micro"):
        run()
        torch.cuda.synchronize()
    cudart.cudaProfilerStop()

    print(json.dumps({"case": args.case, "analytic_read_bytes": rd,
                      "analytic_write_bytes": wr, "warm": warm, "description": desc}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
