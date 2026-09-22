"""T004 step 1 — turn an nsys capture into the timeline tables the task asks for.

From the exports `bench/run_nsys.sh` produces (`cuda_gpu_trace`, `nvtx_gpu_proj_trace`,
`cuda_gpu_kern_sum`) this writes:

  `kernels_per_region.csv`  region, kernel, launches, launches_per_step, total_ms, share
  `gaps.csv`                the gap histogram between consecutive kernels inside each region
  `timeline_summary.csv`    per region: wall time, kernel time (union), GPU idle and its share,
                            kernels, kernels per denoising step

"GPU idle" is region wall time minus the **union** of kernel intervals, so kernels that overlap
are not counted twice. That is the quantity T004's report has to answer (a) with — "kernels
executing vs GPU waiting" — and the same definition E11f uses for `gpu_idle_ae_total`.

Protocol §2: ncu's `Duration` is never a latency. Everything here comes from nsys.

    python bench/parse_nsys.py <run-dir> --tag eager_N4 --steps 4
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json  # noqa: E402


def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def load(nsys_dir: Path, tag: str, kind: str):
    f = next(iter(sorted(glob.glob(str(nsys_dir / f"{tag}*{kind}*.csv")))), None)
    if not f:
        return None, None
    return list(csv.DictReader(open(f))), Path(f).name


def span(r):
    st = num(r.get("Start (ns)") or r.get("Start"))
    du = num(r.get("Duration (ns)") or r.get("Duration"))
    return (st, st + du) if st is not None and du is not None else (None, None)


def union_ms(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    busy, cs, ce = 0.0, ivs[0][0], ivs[0][1]
    for a, b in ivs[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            busy += ce - cs
            cs, ce = a, b
    busy += ce - cs
    return busy / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--inferences", type=int, default=2)
    args = ap.parse_args()
    d = Path(args.rundir)
    nsys = d / "nsys"
    trace, tname = load(nsys, args.tag, "cuda_gpu_trace")
    proj, pname = load(nsys, args.tag, "nvtx_gpu_proj_trace")
    if trace is None:
        print(f"no cuda_gpu_trace export for {args.tag}")
        return 1

    kern = []
    for r in trace:
        a, b = span(r)
        nm = r.get("Name") or ""
        if a is None or not nm:
            continue
        kern.append((a, b, nm))
    kern.sort()
    print(f"  {len(kern)} kernels from {tname}")

    regions = collections.defaultdict(list)
    if proj:
        for r in proj:
            nm = (r.get("Name") or r.get("Range") or "").strip()
            a, b = span(r)
            if a is None or not nm:
                continue
            base = nm.split("[")[0] if "[" in nm else nm
            regions[base].append((a, b))
        print(f"  NVTX ranges from {pname}: "
              + ", ".join(f"{k}x{len(v)}" for k, v in sorted(regions.items())))
    else:
        print("  no nvtx_gpu_proj_trace export — region tables will be empty "
              "(compiled captures run with --no-regions)")

    krows, trows, grows = [], [], []
    scopes = {k: (min(a for a, _ in v), max(b for _, b in v)) for k, v in regions.items()}
    if kern:
        scopes["whole_capture"] = (kern[0][0], max(b for _, b, _ in kern))
    for region, (lo, hi) in sorted(scopes.items()):
        inside = [(max(a, lo), min(b, hi), nm) for a, b, nm in kern if b > lo and a < hi]
        if not inside:
            continue
        wall = (hi - lo) / 1e6
        busy = union_ms([(a, b) for a, b, _ in inside])
        idle = wall - busy
        per_step = len(inside) / (args.steps * args.inferences) if region.startswith("ae") \
            else len(inside) / args.inferences
        trows.append({"region": region, "wall_ms": round(wall, 4),
                      "kernel_union_ms": round(busy, 4), "gpu_idle_ms": round(idle, 4),
                      "gpu_idle_frac": round(idle / wall, 4) if wall else "",
                      "n_kernels": len(inside),
                      "kernels_per_unit": round(per_step, 2),
                      "unit": "denoising step" if region.startswith("ae") else "inference"})
        by = collections.Counter()
        tot = collections.Counter()
        for a, b, nm in inside:
            by[nm] += 1
            tot[nm] += (b - a) / 1e6
        s_tot = sum(tot.values()) or 1.0
        for nm, n in by.most_common(20):
            krows.append({"region": region, "kernel": nm[:120], "launches": n,
                          "launches_per_unit": round(n / max(1, args.inferences), 2),
                          "total_ms": round(tot[nm], 5),
                          "share_of_region_kernel_time": round(tot[nm] / s_tot, 5)})
        seq = sorted((a, b) for a, b, _ in inside)
        gaps = [(seq[i + 1][0] - seq[i][1]) / 1e3 for i in range(len(seq) - 1)
                if seq[i + 1][0] > seq[i][1]]
        if gaps:
            gaps.sort()
            buckets = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100),
                       (100, 1e9)]
            for lo_us, hi_us in buckets:
                c = sum(1 for g in gaps if lo_us <= g < hi_us)
                if c:
                    grows.append({"region": region,
                                  "gap_bucket_us": f"{lo_us}-{'inf' if hi_us > 1e8 else hi_us}",
                                  "count": c,
                                  "total_us": round(sum(g for g in gaps
                                                        if lo_us <= g < hi_us), 2)})
            trows[-1]["gap_total_ms"] = round(sum(gaps) / 1e3, 4)
            trows[-1]["gap_median_us"] = round(gaps[len(gaps) // 2], 3)
            trows[-1]["n_gaps"] = len(gaps)

    for name, rows_ in (("kernels_per_region.csv", krows), ("timeline_summary.csv", trows),
                        ("gaps.csv", grows)):
        if not rows_:
            continue
        keys = sorted({k for r in rows_ for k in r})
        with (d / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in rows_])

    write_json(d / "nsys" / f"{args.tag}_summary.json", {
        "tag": args.tag, "steps": args.steps, "inferences": args.inferences,
        "n_kernels": len(kern), "regions": {k: len(v) for k, v in regions.items()},
        "timeline": trows,
        "notes": [
            "GPU idle is region wall time minus the UNION of kernel intervals, so overlapping "
            "kernels are not double-counted.",
            "All durations are nsys durations. Protocol §2: ncu Duration is never a latency.",
            "Compiled captures run with --no-regions because regions.py's context managers "
            "graph-break the model, so only whole_capture is available for those.",
        ]})
    for t in trows:
        print(f"  {t['region']:14s} wall {t['wall_ms']:9.3f} ms  busy "
              f"{t['kernel_union_ms']:9.3f}  idle {t['gpu_idle_ms']:8.3f} "
              f"({100*(t['gpu_idle_frac'] or 0):5.1f} %)  {t['n_kernels']:5d} kernels")
    print(f"wrote {d}/kernels_per_region.csv, timeline_summary.csv, gaps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
