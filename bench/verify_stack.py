"""T001 step 2 — verify the PyTorch stack: CUDA, bf16 GEMM, torch.compile.

Writes a JSON report; every claim in env/README.md traces back to it.
Usage: python bench/verify_stack.py --out results/<run>/stack_verification.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import summarize, write_json  # noqa: E402


def check_bf16_gemm(torch) -> dict:
    """A bf16 GEMM must run and agree with an fp32 reference."""
    r = {"name": "bf16_gemm_2048"}
    try:
        n = 2048
        torch.manual_seed(0)
        a32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
        a, b = a32.bfloat16(), b32.bfloat16()
        c = a @ b
        torch.cuda.synchronize()
        ref = a32 @ b32
        err = ((c.float() - ref).abs().max() / ref.abs().max()).item()
        r.update({"ok": True, "dtype": str(c.dtype), "shape": list(c.shape),
                  "max_rel_err_vs_fp32": err,
                  "finite": bool(torch.isfinite(c).all().item())})
    except Exception as e:
        r.update({"ok": False, "error": repr(e), "traceback": traceback.format_exc()})
    return r


def check_dtypes(torch) -> dict:
    """Which dtypes multiply at all on this device (peaks come later, in T005/E1)."""
    out = {}
    n = 1024
    for name, dt in [("bfloat16", torch.bfloat16), ("float16", torch.float16),
                     ("float32", torch.float32)]:
        try:
            a = torch.randn(n, n, device="cuda", dtype=dt)
            c = a @ a
            torch.cuda.synchronize()
            out[name] = {"ok": True, "finite": bool(torch.isfinite(c).all().item())}
        except Exception as e:
            out[name] = {"ok": False, "error": repr(e)}
    try:
        a = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
        c = torch._int_mm(a, a)
        torch.cuda.synchronize()
        out["int8__int_mm"] = {"ok": True, "dtype": str(c.dtype)}
    except Exception as e:
        out["int8__int_mm"] = {"ok": False, "error": repr(e)}
    return out


class MLP:
    """Kept as a factory so each compile attempt starts from a fresh module."""

    @staticmethod
    def make(torch):
        import torch.nn as nn
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)
        ).cuda().to(torch.bfloat16).eval()


def try_compile(torch, backend: str, mode: str | None) -> dict:
    """Compile and run a small bf16 MLP. Reports compile time and correctness."""
    label = f"backend={backend}" + (f",mode={mode}" if mode else "")
    r = {"backend": backend, "mode": mode, "label": label}
    try:
        torch._dynamo.reset()
        m = MLP.make(torch)
        x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            eager = m(x).float()
        kw = {"backend": backend}
        if mode:
            kw["mode"] = mode
        cm = torch.compile(m, **kw)
        t0 = time.time()
        with torch.no_grad():
            y = cm(x)
        torch.cuda.synchronize()
        r["compile_and_first_call_s"] = round(time.time() - t0, 3)
        # steady state: no recompilation expected after this
        with torch.no_grad():
            for _ in range(20):
                y = cm(x)
        torch.cuda.synchronize()
        err = ((y.float() - eager).abs().max() / eager.abs().max().clamp(min=1e-6)).item()
        # GPU time of 100 compiled calls, for a sanity figure only (not a protocol result)
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(100)]
        with torch.no_grad():
            for s, e in ev:
                s.record()
                cm(x)
                e.record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s, e in ev]
        r.update({"ok": True, "max_rel_err_vs_eager": err,
                  "finite": bool(torch.isfinite(y).all().item()),
                  "steady_state_ms": summarize(samples),
                  "samples_ms": samples})
    except Exception as e:
        r.update({"ok": False, "error": repr(e),
                  "traceback": traceback.format_exc()[-4000:]})
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-csv", default=None,
                    help="write every per-iteration CUDA-event sample here "
                         "(protocol §3.5: keep every sample, never only the summary)")
    args = ap.parse_args()

    rep = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": sys.argv, "env_vars": {k: v for k, v in os.environ.items()
                                          if k.startswith(("TORCH", "CUDA", "TRITON"))}}
    try:
        import torch
    except Exception as e:
        rep["fatal"] = f"torch import failed: {e!r}"
        write_json(Path(args.out), rep)
        return 1

    rep["torch"] = torch.__version__
    rep["torch_cuda"] = torch.version.cuda
    rep["cuda_available"] = torch.cuda.is_available()
    rep["arch_list"] = torch.cuda.get_arch_list()
    if not torch.cuda.is_available():
        rep["fatal"] = "torch.cuda.is_available() is False"
        write_json(Path(args.out), rep)
        return 1

    p = torch.cuda.get_device_properties(0)
    rep["device"] = {"name": p.name, "sm_count": p.multi_processor_count,
                     "compute_capability": f"{p.major}.{p.minor}",
                     "total_memory_bytes": p.total_memory}
    rep["bf16_gemm"] = check_bf16_gemm(torch)
    rep["dtypes"] = check_dtypes(torch)

    try:
        import triton
        rep["triton"] = triton.__version__
    except Exception as e:
        rep["triton"] = f"absent ({e!r})"

    rep["compile"] = []
    for backend, mode in [("inductor", None),
                          ("inductor", "default"),
                          ("inductor", "reduce-overhead"),
                          ("inductor", "max-autotune"),
                          ("inductor", "max-autotune-no-cudagraphs"),
                          ("aot_eager", None),
                          ("eager", None)]:
        res = try_compile(torch, backend, mode)
        rep["compile"].append(res)
        print(f"  compile {res['label']}: "
              f"{'OK' if res.get('ok') else 'FAILED'}"
              + (f" ({res['compile_and_first_call_s']}s)" if res.get("ok") else ""),
              flush=True)

    if args.samples_csv:
        import csv as _csv
        p_csv = Path(args.samples_csv)
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        with p_csv.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["region", "iter", "ms"])
            for c in rep["compile"]:
                for i, ms in enumerate(c.get("samples_ms", [])):
                    w.writerow([f"compile_probe_mlp:{c['label']}", i, ms])
        print(f"wrote {p_csv}")
    # keep the JSON readable: the samples live in the CSV
    for c in rep["compile"]:
        c.pop("samples_ms", None)

    working = [c["label"] for c in rep["compile"] if c.get("ok")]
    rep["compile_working"] = working
    rep["torch_compile_inductor_works"] = any(
        c.get("ok") and c["backend"] == "inductor" for c in rep["compile"])
    write_json(Path(args.out), rep)
    print(json.dumps({"cuda": rep["cuda_available"],
                      "bf16_gemm_ok": rep["bf16_gemm"].get("ok"),
                      "inductor_works": rep["torch_compile_inductor_works"],
                      "working": working}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
