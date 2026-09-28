"""Where the action expert's time goes, by model component — from the committed E5 nsys runs (no new measurement).

For each NVTX range whose name matches --range (e.g. 'ae_step[' for the LeRobot experts, 'ae_total' for GR00T), the
CUDA runtime calls issued inside it are joined to the GPU kernels they launched (correlationId; CUDA-graph replays
included). Each kernel is assigned to a component by its name; per range we report kernel time per component, the GPU
span (first kernel start -> last kernel end) and the idle time inside it (span minus the union of kernel intervals).

Component rules (name substrings, checked in order):
  gemm           cuBLAS / CUTLASS GEMM kernels ('gemm', 'cutlass', 'Kernel2') and Inductor GEMM templates
                 ('triton_tem*', '*addmm*': the matmul with its fused epilogue)  -- projections AND attention products
                 (the two are separated by grid size: attention bmm have gridZ = heads > 1 or come from 'bmm' ops;
                 we report gemm_batched (gridZ > 1) apart from gemm_2d)
  attention      flash / fmha / sdpa kernels, softmax
  rope           triton kernels fusing cos/sin/neg (RoPE)
  norm           rsqrt / mean / layer_norm / rms kernels
  act_mlp        gelu / silu / mul fusions (GeGLU)
  copy_cat_kv    cat / clone / copy / index / scatter / slice kernels (K/V cache append, casts)
  elementwise    remaining elementwise / reduce kernels
  other          anything else

    python bench/e5_expert_components.py <nsys.sqlite> <range-prefix> <out.csv> [label]
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict


def component(name: str, gz: int) -> str:
    n = name.lower()
    if n.startswith("triton_tem") or "addmm" in n or "_mm_" in n or n.startswith("triton_mm"):
        # Inductor GEMM template (forced autotune): the matmul with its epilogue (bias, GELU, LayerNorm...) fused in
        return "gemm_2d"
    if "fmha" in n or "flash_fwd" in n:
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "kernel2", "gemv")):
        return "gemm_batched" if gz > 1 else "gemm_2d"
    if any(k in n for k in ("flash", "fmha", "sdpa", "softmax", "attention")):
        return "attention"
    if "cos" in n and ("sin" in n or "neg" in n):
        return "rope"
    if any(k in n for k in ("rsqrt", "layer_norm", "rms", "_norm")):
        return "norm"
    if any(k in n for k in ("gelu", "silu", "tanh")):
        return "act_mlp"
    if any(k in n for k in ("cat", "clone", "copy", "index", "scatter", "slice", "gather", "fill")):
        return "copy_cat_kv"
    if any(k in n for k in ("elementwise", "reduce", "triton", "vectorized", "unrolled", "add", "mul")):
        return "elementwise"
    return "other"


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def main() -> int:
    db, rng, out = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else db
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    ranges = [r for r in c.execute("select text, start, end, globalTid from NVTX_EVENTS "
                                   "where text like ? order by start", (rng + "%",))]
    kern = c.execute("select start, end, correlationId, demangledName, gridZ from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    by_corr = defaultdict(list)
    for s, e, cid, nm, gz in kern:
        by_corr[cid].append((s, e, names.get(nm, str(nm)), gz or 1))
    rt = c.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    rows = []
    for text, s, e, tid in ranges:
        ks = [k for st, cid, t in rt if s <= st <= e and t == tid for k in by_corr.get(cid, [])]
        if not ks:
            continue
        per = defaultdict(float)
        cnt = defaultdict(int)
        for ks_, ke, nm, gz in ks:
            comp = component(nm, gz)
            per[comp] += (ke - ks_) / 1e6
            cnt[comp] += 1
        span = (max(k[1] for k in ks) - min(k[0] for k in ks)) / 1e6
        busy = union([(k[0], k[1]) for k in ks]) / 1e6
        rows.append({"label": label, "range": text, "kernels": len(ks), "span_ms": round(span, 4),
                     "busy_ms": round(busy, 4), "idle_ms": round(span - busy, 4),
                     **{f"{k}_ms": round(v, 4) for k, v in per.items()},
                     **{f"{k}_n": v for k, v in cnt.items()}})
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("label", "range", "kernels", "span_ms", "busy_ms", "idle_ms"), k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, 0) for k in keys} for r in rows])
    # mean over ranges
    n = len(rows)
    comps = sorted({k[:-3] for k in keys if k.endswith("_ms") and k not in ("span_ms", "busy_ms", "idle_ms")})
    span = sum(r["span_ms"] for r in rows) / n
    print(f"{label}: {n} ranges '{rng}*', mean span {span:.2f} ms, busy {sum(r['busy_ms'] for r in rows)/n:.2f}, "
          f"idle {sum(r['idle_ms'] for r in rows)/n:.2f}, kernels {sum(r['kernels'] for r in rows)/n:.1f}")
    for cp in comps:
        ms = sum(r.get(cp + "_ms", 0) for r in rows) / n
        kn = sum(r.get(cp + "_n", 0) for r in rows) / n
        print(f"   {cp:14s} {ms:8.3f} ms  ({100*ms/span:5.1f} % of span)  {kn:6.1f} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
