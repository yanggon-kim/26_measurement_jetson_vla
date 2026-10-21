"""T005 — turn the four ncu runs into `ncu/sysmem_miss_validation.csv`.

Answers protocol §3.3 question 1: does `lts__t_sectors_aperture_sysmem_*_lookup_miss.sum x 32 B`
recover the off-chip traffic of a kernel whose traffic is known by construction?

    python bench/parse_ncu_validation.py results/<run>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SECTOR = 32
CASES = ["copy_256mib", "gemm_cold", "gemm_warm", "gemm_l2", "gemm_l2_twice"]
M_RD = "lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum"
M_WR = "lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum"
M_ALL = "lts__t_sectors_aperture_sysmem_lookup_miss.sum"
M_RD_ALL = "lts__t_sectors_aperture_sysmem_op_read.sum"
M_WR_ALL = "lts__t_sectors_aperture_sysmem_op_write.sum"
M_HIT = "lts__t_sectors_aperture_sysmem_lookup_hit.sum"
M_DUR = "gpu__time_duration.sum"
M_SM = "gpc__cycles_elapsed.avg.per_second"


def read_case(d: Path, case: str) -> dict | None:
    f = d / "ncu" / f"{case}.csv"
    if not f.exists():
        return None
    lines = [ln for ln in f.read_text(errors="replace").splitlines()
             if not ln.startswith("==")]
    rows = list(csv.DictReader(lines))
    # ncu emits a units row right after the header; the data row is the one with digits
    data = [r for r in rows if (r.get(M_DUR) or "").replace(".", "").replace(",", "").isdigit()]
    if not data:
        data = [r for r in rows if r.get("Kernel Name")]
    if not data:
        return None
    r = data[-1]

    def num(k):
        v = (r.get(k) or "").replace(",", "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    return {"kernel_name": r.get("Kernel Name", ""), "raw": r,
            "rd": num(M_RD), "wr": num(M_WR), "all": num(M_ALL),
            "rd_all": num(M_RD_ALL), "wr_all": num(M_WR_ALL), "hit": num(M_HIT),
            "dur": num(M_DUR), "sm_hz": num(M_SM)}


def main() -> int:
    d = Path(sys.argv[1])
    peaks = json.loads((Path(__file__).resolve().parent.parent / "results"
                        / "20260921_E1_peak" / "hw_constants.json").read_text())
    peak_bw = peaks["measured_peaks"]["peak_memory_bandwidth_GBps"]

    out = []
    for case in CASES:
        an = json.loads((d / "raw" / f"ncu_{case}_analytic.json").read_text())
        c = read_case(d, case)
        if c is None:
            out.append({"kernel": case, "note": "no ncu data"})
            continue
        # ncu reports gpu__time_duration.sum in ns on this build
        ms = (c["dur"] or 0) / 1e6
        mrd = (c["rd"] or 0) * SECTOR
        mwr = (c["wr"] or 0) * SECTOR
        mall = (c["all"] or 0) * SECTOR
        ard, awr = an["analytic_read_bytes"], an["analytic_write_bytes"]
        sec = ms / 1e3 if ms else None
        out.append({
            "kernel": case,
            "kernel_name": c["kernel_name"][:60],
            "cache_control": "none (warm)" if an["warm"] else "default (flushed, cold)",
            "analytic_read_bytes": ard, "analytic_write_bytes": awr,
            "miss_read_sectors": int(c["rd"] or 0),
            "miss_write_sectors": int(c["wr"] or 0),
            "miss_all_sectors": int(c["all"] or 0),
            "hit_sectors": int(c["hit"] or 0),
            "total_read_sectors": int(c["rd_all"] or 0),
            "total_write_sectors": int(c["wr_all"] or 0),
            "measured_read_bytes": int(mrd), "measured_write_bytes": int(mwr),
            "ratio_read": round(mrd / ard, 4) if ard else "",
            "ratio_write": round(mwr / awr, 4) if awr else "",
            # `op_write_lookup_miss` is identically zero on this device: a write sector is
            # recorded as a lookup HIT (the L2 is write-back / write-allocate), so the miss
            # counter never sees it. The *total* write-sector counter does recover the
            # traffic, so the write substitute has to be op_write.sum, not its miss variant.
            "measured_write_bytes_from_op_write": int((c["wr_all"] or 0) * SECTOR),
            "ratio_write_from_op_write": round((c["wr_all"] or 0) * SECTOR / awr, 4)
                                          if awr else "",
            # protocol §3.3(3) and §2: ncu's Duration is NOT a latency -- ncu serialises and
            # flushes caches. It is recorded for reference under its real name, and the
            # bandwidth columns it would produce are deliberately left blank: they would need
            # an nsys duration of the same kernel, and no nsys run exists in this directory.
            "ncu_duration_ms": round(ms, 6),
            "GBps_from_misses": "",
            "GBps_analytic": "",
            "pct_of_measured_peak": "",
            "GBps_columns_blank_because": "protocol §3.3(3): achieved GB/s must use an nsys "
                                          "duration, never ncu Duration; no nsys run here",
            # protocol §3.2: ncu's own SM clock, and how far it is from the pinned value
            "sm_clock_hz_from_ncu": int(c["sm_hz"]) if c["sm_hz"] else "",
            "sm_clock_pct_diff_vs_pinned": round(
                100 * (c["sm_hz"] - 1300500000) / 1300500000, 2) if c["sm_hz"] else "",
            "sm_clock_within_2pct": (abs(c["sm_hz"] - 1300500000) / 1300500000 < 0.02)
                                     if c["sm_hz"] else "",
            "description": an["description"],
        })

    f = d / "ncu" / "sysmem_miss_validation.csv"
    with f.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{'case':14s} {'a_read_MB':>10s} {'m_read_MB':>10s} {'r_read':>7s} "
          f"{'a_wr_MB':>8s} {'wr_op_MB':>9s} {'r_wr_op':>8s} {'ncu_ms':>8s} "
          f"{'SM_MHz':>8s} {'d%':>6s}")
    for r in out:
        if "analytic_read_bytes" not in r:
            print(f"{r['kernel']:14s} {r.get('note')}")
            continue
        print(f"{r['kernel']:14s} {r['analytic_read_bytes']/1e6:10.2f} "
              f"{r['measured_read_bytes']/1e6:10.2f} {r['ratio_read']:7} "
              f"{r['analytic_write_bytes']/1e6:8.2f} "
              f"{r['measured_write_bytes_from_op_write']/1e6:9.2f} "
              f"{r['ratio_write_from_op_write']:8} {r['ncu_duration_ms']:8.4f} "
              f"{(r['sm_clock_hz_from_ncu'] or 0)/1e6:8.1f} "
              f"{r['sm_clock_pct_diff_vs_pinned']:6}")
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
