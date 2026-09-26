"""S-sweep per-call GEMM time from the ncu kernel durations, replacing the CUDA-event column.

bench/micro_proj_sweep.py (up to 2026-09-25) timed each F.linear with its OWN event pair and a
synchronize after every call. On Orin that adds ~45-75 us per call: SmolVLA's q_proj reads
89.6 us under events and 16.5 us as a kernel under ncu; every small GEMM in all three sweeps
sat at ~89-97 us whatever its shape. The events measured the timing harness, not the GEMM.

The ncu captures of the same sweep (one per projection x S, `-c 3` kernels, recorded in the
same session) hold each kernel's gpu__time_duration. Per call = mean duration of the GEMM
kernel + mean duration of cuBLASLt's splitKreduce kernel when the call launched one (a split-K
GEMM is two kernels; `-c 3` then holds main, reduce, main). Bytes / FLOPs / floor are the
sweep's own (s_sweep.csv), only the time is replaced.

    python bench/sweep_ncu_table.py results/<date>_E5b_<model>_ssweep   ->  s_sweep_ncu.csv
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

UNIT_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}
MEASURED_GBPS = 187.47


def per_capture(det: Path) -> dict:
    kern, seen = [], set()
    metrics: dict[str, dict] = {}
    for r in csv.DictReader(open(det)):
        i = r["ID"]
        metrics.setdefault(i, {})[r["Metric Name"]] = r["Metric Value"]
        if r["Metric Name"] == "Duration" and i not in seen:
            seen.add(i)
            kern.append((i, r["Kernel Name"],
                         float(r["Metric Value"].replace(",", "")) * UNIT_US[r["Metric Unit"]]))
    red = [d for _, n, d in kern if "splitKreduce" in n]
    main = [(i, n, d) for i, n, d in kern if "splitKreduce" not in n]
    if not main:
        raise SystemExit(f"{det}: no GEMM kernel")
    m0 = metrics[main[0][0]]
    return {"kernel": main[0][1][:90], "n_main": len(main), "n_reduce": len(red),
            "main_us": statistics.fmean(d for _, _, d in main),
            "reduce_us": statistics.fmean(red) if red else 0.0,
            "sm_pct": m0.get("Compute (SM) Throughput", ""),
            "mem_pct": m0.get("Memory Throughput", "")}


def main() -> int:
    d = Path(sys.argv[1])
    ev = {(r["proj"], r["S"]): r for r in csv.DictReader(open(d / "s_sweep.csv"))}
    rows = []
    for (proj, S), r in sorted(ev.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        det = d / "ncu" / f"sweep_{proj}_S{S}_details.csv"
        if not det.exists():
            rows.append({"proj": proj, "S": S, "note": "no ncu capture"})
            continue
        c = per_capture(det)
        us = c["main_us"] + c["reduce_us"]
        nbytes, floor = float(r["bytes"]), float(r["floor_us"])
        rows.append({
            "model": r["model"], "proj": proj, "proj_name": r["proj_name"], "in": r["in"],
            "out": r["out"], "dtype": r["dtype"], "S": S,
            "per_call_us_ncu": round(us, 3), "main_kernel_us": round(c["main_us"], 3),
            "splitK_reduce_us": round(c["reduce_us"], 3), "kernel": c["kernel"],
            "floor_us": floor, "x_floor_ncu": round(us / floor, 3),
            "achieved_GBps_ncu": round(nbytes / (us * 1e-6) / 1e9, 2),
            "pct_of_measured_bw": round(100 * nbytes / (us * 1e-6) / 1e9 / MEASURED_GBPS, 1),
            "sm_throughput_pct": c["sm_pct"], "memory_throughput_pct": c["mem_pct"],
            "event_median_us_SUPERSEDED": r["median_us"],
            "event_overhead_us": round(float(r["median_us"]) - us, 2)})
    out = d / "s_sweep_ncu.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['proj']:13s} S={r['S']:>4}  ncu {r.get('per_call_us_ncu', '-'):>8} us  "
              f"x_floor {r.get('x_floor_ncu', '-'):>6}  (events said {r.get('event_median_us_SUPERSEDED', '-')})")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
