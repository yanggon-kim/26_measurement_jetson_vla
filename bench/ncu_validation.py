"""T003 step 3: validate the DRAM-byte and FLOP counters (protocol §3.3) on kernels with known traffic and work.

One case per process; the case's kernel(s) run once inside NVTX range "val" between cudaProfilerStart/Stop after a
warm-up. ncu (form D, explicit metric list, default cache control = caches flushed before each launch, i.e. cold) is
run over it by the driver loop in the T003 commands; bench/parse_ncu_validation.py turns the exports into
validation.csv. Prints the case's analytic read / write bytes and FLOPs as JSON.

Byte cases (streaming, no reuse, working set >> the 32 MiB L2):
  copy_sm      z = 1.0 * x, 512 MiB bf16   (an SM kernel; d2d copy_ runs on the copy engine, invisible to ncu)
  axpy         y += a * x, 512 MiB each    (read x, read y, write y)
  read_sum     x.sum(), 512 MiB            (read only)
  gemv         [1, 8192] x [8192, 8192]^T bf16 (reads the 128 MiB weight once)
FLOP cases (4096^3, every dtype; analytic FLOPs = 2 n^3):
  gemm_fp32 (highest, SIMT) gemm_tf32 gemm_bf16 gemm_fp16 gemm_int8 gemm_fp8
plus skinny_bf16: the T001 toy linear x[51, 1024] W[4096, 1024] (tile padding of M shows in the tensor counter).

~/venvs/base/bin/python bench/ncu_validation.py --case gemv
"""
import argparse
import json

import torch
import torch.nn.functional as F

MiB = 1 << 20


def build(case):
    dev = "cuda"
    torch.manual_seed(0)
    if case == "copy_sm":
        n = 512 * MiB // 2
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        z = torch.empty_like(x)
        return (lambda: torch.mul(x, 1.0, out=z)), {"read": 2 * n, "write": 2 * n, "flops": 0, "dtype": "bf16"}
    if case == "axpy":
        n = 512 * MiB // 2
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        y = torch.randn(n, device=dev, dtype=torch.bfloat16)
        return (lambda: y.add_(x, alpha=1.0001)), {"read": 4 * n, "write": 2 * n, "flops": 2 * n, "dtype": "bf16",
                                                    "flops_note": "1 FMA per element counted as 2 FLOPs; bf16 math may run as fp32 SIMT after conversion"}
    if case == "read_sum":
        n = 512 * MiB // 2
        x = torch.randn(n, device=dev, dtype=torch.bfloat16)
        return (lambda: x.sum()), {"read": 2 * n, "write": 4, "flops": n, "dtype": "bf16",
                                   "flops_note": "one add per element (fp32 accumulate)"}
    if case == "gemv":
        k = 8192
        x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(k, k, device=dev, dtype=torch.bfloat16)
        return (lambda: F.linear(x, w)), {"read": 2 * k * k + 2 * k, "write": 2 * k, "flops": 2 * k * k, "dtype": "bf16"}
    if case == "skinny_bf16":
        x = torch.randn(51, 1024, device=dev, dtype=torch.bfloat16)
        w = torch.randn(4096, 1024, device=dev, dtype=torch.bfloat16)
        return (lambda: F.linear(x, w)), {"read": 2 * 4096 * 1024 + 2 * 51 * 1024, "write": 2 * 51 * 4096,
                                          "flops": 2 * 51 * 4096 * 1024, "dtype": "bf16",
                                          "flops_note": "useful FLOPs; the kernel's tile pads M"}
    if case.startswith("gemm_"):
        n = 4096
        name = case[5:]
        torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
        torch.set_float32_matmul_precision("high" if name == "tf32" else "highest")
        if name == "int8":
            a = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
            b = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
            fn, esz, osz = (lambda: torch._int_mm(a, b)), 1, 4
        elif name == "fp8":
            a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()
            one = torch.ones((), device=dev)
            fn, esz, osz = (lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)), 1, 2
        else:
            dt = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
            fn, esz, osz = (lambda: a @ b), a.element_size(), a.element_size()
        return fn, {"read": 2 * n * n * esz, "write": n * n * osz, "flops": 2 * n ** 3, "dtype": name,
                    "bytes_note": "read = A + B once (a lower bound: tiles re-read operands that do not stay in L2)"}
    raise SystemExit(f"unknown case {case}")


CASES = ["copy_sm", "axpy", "read_sum", "gemv", "skinny_bf16", "gemm_fp32", "gemm_tf32", "gemm_bf16", "gemm_fp16",
         "gemm_int8", "gemm_fp8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    a = ap.parse_args()
    fn, info = build(a.case)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("val"):
        fn()
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    info["case"] = a.case
    print("ANALYTIC " + json.dumps(info))


if __name__ == "__main__":
    main()
