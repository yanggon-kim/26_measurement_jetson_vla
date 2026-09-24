"""Does GEMM autotuning fix π0's mio_throttle-bound prefix GEMM? Forced, and measured.

Job 2 found that π0's compiled `vlm_prefix` spends 47.3 % of its captured window in two
kernels at grid (16, 7, 9) running at 25.9 % SM and 36.9 % memory throughput, stalled 48 %
on `mio_throttle` — while the SAME kernel at grid (128, 7, 1) reaches 85.6 % SM on the same
dtype. That is a tiling problem, not a hardware limit, so the obvious lever is to let
inductor autotune the GEMM.

**It is disabled on this device and says so quietly.** `torch/_inductor/utils.py:1426`,
`is_big_gpu()`:

    min_sms = 16 if device.type == "xpu" else 68  # 3080
    if avail_sms < min_sms:
        log.warning("Not enough SMs to use max_autotune_gemm mode")
        return False

Orin has **16** SMs against a required 68, so `mode="max-autotune"` silently keeps cuBLAS
for every GEMM and autotunes only pointwise/reduction Triton kernels. That warning is
already in the Part B logs; it was never acted on.

This script forces `is_big_gpu()` to True so Triton GEMM templates are actually
benchmarked, and measures the prefix against the two references we already have. If the
forced autotune does not beat cuBLAS, the answer is that cuBLAS's tiling is the best
available here and the mio_throttle kernel needs a different fix — which is a real result
either way.

    python bench/try_autotune_prefix.py --mode forced_autotune
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0")
    ap.add_argument("--mode", default="forced_autotune",
                    choices=("eager", "cublas_autotune", "forced_autotune"))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--outdir", default="results/20260923_E4i_pi0_prefix_autotune")
    args = ap.parse_args()

    forced = False
    if args.mode == "forced_autotune":
        # Override the SM-count gate so Triton GEMM templates are actually benchmarked.
        # This is the whole point of the experiment; it is recorded in meta.json so no one
        # mistakes these numbers for a stock configuration.
        import torch._inductor.config as IC
        import torch._inductor.utils as IU
        IU.is_big_gpu = lambda *a, **k: True
        # Restrict the backends to TRITON. Leaving ATEN in the list while forcing the gate
        # makes inductor register aten.bmm as BOTH a fallback and a decomposition and it
        # asserts: "both a fallback and a decomp for same op: aten.bmm.default". The tables
        # are built conditionally on the same gate this override flips.
        for k, v in (("max_autotune", True), ("max_autotune_gemm", True),
                     ("max_autotune_gemm_backends", "TRITON")):
            if hasattr(IC, k):
                setattr(IC, k, v)
        forced = True

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    spec = MR.get(args.model)
    policy, _, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    torch.set_float32_matmul_precision("high")

    desc = "eager"
    if args.mode != "eager":
        pg = P_container = getattr(policy.model, spec.container)
        t0 = time.time()
        pg.forward = torch.compile(pg.forward, mode="max-autotune-no-cudagraphs")
        desc = ("max-autotune-no-cudagraphs on pg.forward, is_big_gpu FORCED True"
                if forced else "max-autotune-no-cudagraphs on pg.forward, stock")
        print(f"  compiling: {desc} ({time.time()-t0:.1f} s to call)", flush=True)

    rec = R.Recorder(enable_layers=False, layers_in_step=0)
    undo = R.instrument(policy, rec, spec.container)

    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    t0 = time.time()
    for _ in range(args.warmup):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    print(f"  warm-up (includes autotuning): {time.time()-t0:.1f} s", flush=True)

    e2e, per = [], []
    for _ in range(args.timed):
        rec.reset()
        rec.start()
        t1 = time.perf_counter()
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t1) * 1e3)
        rec.stop()
        per.append(rec.results_ms())
    undo()

    med = lambda xs: statistics.median(xs)
    out = {
        "model": args.model, "mode": args.mode, "compile": desc,
        "is_big_gpu_forced": forced,
        "N": args.steps, "n_timed": args.timed,
        "e2e_median_ms": round(med(e2e), 3),
        "vision_ms": round(med([p.get("vision", 0.0) for p in per]), 3),
        "vlm_prefix_ms": round(med([p.get("vlm_prefix", 0.0) for p in per]), 3),
        "ae_total_ms": round(med([p.get("ae_total", 0.0) for p in per]), 3),
        "reference_eager_prefix_ms": 234.57,
        "reference_vlm_compile_prefix_ms": 192.76,
        "note": ("inductor's is_big_gpu() requires 68 SMs and this device has 16, so "
                 "mode='max-autotune' keeps cuBLAS for GEMMs unless the gate is forced. "
                 "forced_autotune overrides it; cublas_autotune is the stock behaviour."),
        "gate": gate,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
    }
    d = Path(args.outdir)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.mode}.json").write_text(json.dumps(out, indent=1))
    print(f"\n  e2e {out['e2e_median_ms']} ms | vision {out['vision_ms']} | "
          f"PREFIX {out['vlm_prefix_ms']} | ae {out['ae_total_ms']}")
    print(f"  references: eager prefix 234.57, vlm_compile prefix 192.76")
    print(f"  wrote {d}/{args.mode}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
