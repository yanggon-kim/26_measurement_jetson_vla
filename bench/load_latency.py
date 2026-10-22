"""T003 step 4 (optional): dependent-load latency by pointer chasing, per working-set size.

One thread of one block follows a random single cycle through a buffer of 128-byte-spaced 32-bit indices (each hop
lands on a new cache line; the random order defeats prefetching).
  working set ≤ L2: --hops dependent loads after a warm pass (many passes over the cycle: L1 / L2 hit latency).
  working set > L2: before every repeat the L2 is flushed by writing a 256 MiB buffer, the walk starts at a random
                    line and is at most one full cycle long (no line visited twice), so every load misses L2.
                    (First version: every repeat restarted at the same line and walked 200,000 lines = 25.6 MB,
                    which fits the 32 MiB L2, so repeats 1-4 partly hit L2 -- T003 review.) Measured per hop:
SM cycles (clock64 inside the kernel) and ns (CUDA events around the kernel ÷ hops). Sizes span L1, L2 (32 MiB) and
DRAM. The kernel is compiled on the device with torch.utils.cpp_extension.load_inline (nvcc 13.0, sm_110).

python bench/load_latency.py --out results/<run>/load_latency.csv
"""
import argparse
import csv
import os
import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402

CUDA_SRC = r"""
#include <torch/extension.h>
__global__ void chase(const unsigned* __restrict__ next, long hops, unsigned start, unsigned* out, long long* cyc) {
    unsigned p = start;
    long long t0 = clock64();
    for (long i = 0; i < hops; ++i) p = next[p];
    long long t1 = clock64();
    out[0] = p;            // keeps the chain live
    cyc[0] = t1 - t0;
}
std::vector<torch::Tensor> run(torch::Tensor next, long hops, long start) {
    auto out = torch::zeros({1}, next.options());
    auto cyc = torch::zeros({1}, next.options().dtype(torch::kInt64));
    chase<<<1, 1>>>((const unsigned*)next.data_ptr<int>(), hops, (unsigned)start, (unsigned*)out.data_ptr<int>(),
                    (long long*)cyc.data_ptr<int64_t>());
    return {out, cyc};
}
"""
CPP_SRC = "std::vector<torch::Tensor> run(torch::Tensor next, long hops, long start);"


def build_chain(nbytes, stride_b=128, seed=0):
    """A single random cycle over the cache lines of an nbytes buffer, stored as int32 indices."""
    n_int = nbytes // 4
    step = stride_b // 4
    lines = torch.arange(0, n_int, step, dtype=torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = lines[torch.randperm(len(lines), generator=g)]
    nxt = torch.zeros(n_int, dtype=torch.int32)
    nxt[perm] = torch.roll(perm, -1).to(torch.int32)
    return nxt.cuda(), int(perm[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hops", type=int, default=200000)
    ap.add_argument("--sizes-kib", default="16,64,256,1024,4096,16384,24576,65536,262144,1048576")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "11.0")
    mod = load_inline("ptrchase", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["run"],
                      extra_cuda_cflags=["-O3"], verbose=False,
                      build_directory=os.environ.get("PTRCHASE_BUILD", None))
    g0 = gate.run_gate()
    if not g0["gate_pass"]:
        raise SystemExit(f"gate failed {g0['checks']}")
    rows = []
    L2 = 33554432
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    g = torch.Generator().manual_seed(1)
    for kib in [int(x) for x in a.sizes_kib.split(",")]:
        nxt, start = build_chain(kib * 1024)
        lines = kib * 1024 // 128
        beyond = kib * 1024 > L2
        hops = min(lines, 1_000_000) if beyond else a.hops
        mod.run(nxt, min(lines, 10000), start)          # touches pages / TLB once
        torch.cuda.synchronize()
        cpu_next = nxt.cpu() if beyond else None
        for r in range(a.reps):
            st = start
            if beyond:
                st = int(cpu_next[int(torch.randint(0, lines, (1,), generator=g)) * 32])   # a random line of the cycle
                flush.fill_(r & 0xFF)
                torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out, cyc = mod.run(nxt, hops, st)
            e.record()
            e.synchronize()
            ms = s.elapsed_time(e)
            rows.append({"size_KiB": kib, "rep": r, "hops": hops, "cycles_per_load": round(cyc.item() / hops, 2),
                         "ns_per_load": round(ms * 1e6 / hops, 2), "kernel_ms": round(ms, 4),
                         "lines_in_cycle": lines, "l2_flushed_cold_walk": beyond})
        print(kib, "KiB:", rows[-1]["cycles_per_load"], "cycles,", rows[-1]["ns_per_load"], "ns", flush=True)
        del nxt
        torch.cuda.empty_cache()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
