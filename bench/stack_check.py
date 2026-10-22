"""T001 step 3: verify a PyTorch route on this device (docs/05_setup.md A8).

Smoke checks, not measurements: CUDA available, device arch in the wheel's arch list, a bf16 GEMM 4096^3
against an fp32 reference, torch.compile of a bf16 MLP (1024 -> 4096 -> 1024, GeLU, x [51, 1024]) in four
modes (20 warm-up, 100 timed CUDA-event calls each, max relative error vs eager), triton import, and
Inductor's is_big_gpu SM threshold. Writes one JSON to --out.

Usage: python bench/stack_check.py --out results/<run>/stack_verification.json
"""
import argparse
import inspect
import json
import logging
import platform
import re
import statistics
import time
import traceback

import torch


def rel_err(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm()).item()


def check_gemm():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    ref = torch.matmul(a.double(), b.double()).float()
    out = torch.matmul(a.bfloat16(), b.bfloat16())
    torch.cuda.synchronize()
    err = rel_err(out, ref)
    # bf16 has an 8-bit significand: input rounding alone gives ~2^-9 relative error per element; a K = 4096
    # dot product of independent rounding errors stays near that level in the norm-wise relative error.
    return {"shape": [4096, 4096, 4096], "rel_err_vs_fp64_ref": err, "tolerance": 1e-2, "pass": err < 1e-2}


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(1024, 4096)
        self.down = torch.nn.Linear(4096, 1024)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


def time_calls(fn, x, n_warm=20, n_timed=100):
    for _ in range(n_warm):
        fn(x)
    torch.cuda.synchronize()
    ms = []
    for _ in range(n_timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(x)
        e.record()
        e.synchronize()
        ms.append(s.elapsed_time(e))
    return ms


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def check_compile():
    torch.manual_seed(0)
    m = MLP().cuda().bfloat16().eval()
    x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
    out = {}
    with torch.no_grad():
        ref = m(x).clone()
        eager_ms = time_calls(m, x)
        out["eager"] = {"median_ms": statistics.median(eager_ms), "n": len(eager_ms)}
        cap = _Capture()
        logging.getLogger("torch._inductor").addHandler(cap)
        for mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
            torch._dynamo.reset()
            cap.msgs.clear()
            rec = {"mode": mode}
            try:
                f = torch.compile(m, mode=mode)
                t0 = time.perf_counter()
                y = f(x)
                torch.cuda.synchronize()
                rec["first_call_s"] = time.perf_counter() - t0
                ms = time_calls(f, x)
                y = f(x).clone()
                rec.update(
                    median_ms=statistics.median(ms),
                    min_ms=min(ms),
                    n=len(ms),
                    max_rel_err_vs_eager=rel_err(y, ref),
                    finite=bool(torch.isfinite(y).all()),
                    not_enough_sms_logged=any("Not enough SMs" in s for s in cap.msgs),
                    ok=True,
                )
            except Exception as exc:  # recorded, not hidden
                rec.update(ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
            out[mode] = rec
        logging.getLogger("torch._inductor").removeHandler(cap)
    return out


def is_big_gpu_info():
    from torch._inductor import utils as iu

    src = inspect.getsource(iu.is_big_gpu)
    m = re.search(r"min_sms\s*=.*?else\s*(\d+)", src)
    return {
        "min_sms_threshold": int(m.group(1)) if m else "n/a: pattern not found",
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "is_big_gpu": bool(iu.is_big_gpu(0)),
        "source_file": iu.__file__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    res = {
        "label": "smoke check, not a measurement",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": p.name,
        "capability": list(torch.cuda.get_device_capability()),
        "arch_list": torch.cuda.get_arch_list(),
        "sm_count": p.multi_processor_count,
        "l2_bytes": getattr(p, "L2_cache_size", "n/a"),
        "total_memory_bytes": p.total_memory,
    }
    cap = "sm_%d%d" % tuple(res["capability"])
    res["arch_in_list"] = cap in res["arch_list"]
    try:
        import triton

        res["triton"] = triton.__version__
    except Exception as exc:
        res["triton"] = f"import failed: {exc}"
    try:
        import torchvision

        res["torchvision"] = torchvision.__version__
    except Exception as exc:
        res["torchvision"] = f"import failed: {exc}"
    res["bf16_gemm"] = check_gemm()
    res["is_big_gpu"] = is_big_gpu_info()
    res["compile"] = check_compile()
    res["all_pass"] = bool(
        res["cuda_available"]
        and res["arch_in_list"]
        and res["bf16_gemm"]["pass"]
        and not str(res["triton"]).startswith("import failed")
        and all(res["compile"][m].get("ok") and res["compile"][m].get("max_rel_err_vs_eager", 1) < 2e-2
                for m in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    )
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "compile"}, indent=2))
    for m, r in res["compile"].items():
        print(m, {k: v for k, v in r.items() if k != "traceback"})


if __name__ == "__main__":
    main()
