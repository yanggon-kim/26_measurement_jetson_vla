"""Measured DRAM bytes per frame for the E12b prefix and expert vs E6's bytes and the EMC-implied bytes.

Input: <outdir>/ncu/e12b_dram_raw.csv (ncu --page raw --csv of bench/run_ncu_e12b_dram.sh).
Output: <outdir>/dram_bytes.csv, <outdir>/summary.json, <outdir>/meta.json.

Three byte counts per frame, per stage, side by side:
  measured   sum over every kernel of the NVTX range of dram__bytes_read.sum + dram__bytes_write.sum,
             divided by the range's frame count (5, bench/e12b_nsys.py)
  E6         analytic bytes of the matched rung (results/20260925_E6_rungs/pi0_bf16_both/roofline.csv):
             prefix = vision + vlm_prefix, expert = N x ae_step
  EMC        E12b's calibrated EMC draw (reports/E12b_pi0_compiled_corun.md, whole GPU) x the
             unprofiled stage time (results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv,
             graphs_both full_gpu A_iso / B_iso)
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FRAMES = {"prefix_alone": 5, "expert_alone": 5}
N = 10
UNIT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9,
        "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "ns": 1e-9, "us": 1e-6, "ms": 1e-3}


def e6_bytes() -> dict:
    rows = {r["region_or_kernel"]: float(r["bytes"]) for r in csv.DictReader(
        open(REPO / "results/20260925_E6_rungs/pi0_bf16_both/roofline.csv"))}
    return {"prefix_alone": rows["vision"] + rows["vlm_prefix"], "expert_alone": N * rows["ae_step"]}


def stage_ms() -> dict:
    for r in csv.DictReader(open(REPO / "results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv")):
        if r["split"] == "full_gpu":
            return {"prefix_alone": float(r["A_iso_ms"]), "expert_alone": float(r["B_iso_ms"])}
    raise SystemExit("no full_gpu row")


EMC_GBPS = {"prefix_alone": 84.6, "expert_alone": 80.6}   # E12b report, whole GPU, calibrated


def main() -> int:
    out = Path(sys.argv[1])
    raw = out / "ncu" / "e12b_dram_raw.csv"
    rows = list(csv.reader(open(raw)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(hdr)}
    nv = next(i for i, h in enumerate(hdr) if "Push/Pop_Range" in h or h.startswith("thread Domain"))
    need = ["dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum"]
    agg = {k: {"kernels": 0, "read": 0.0, "write": 0.0, "dur_s": 0.0} for k in FRAMES}
    unattributed = 0
    for r in data:
        rng = next((k for k in FRAMES if k in r[nv]), None)
        if rng is None:
            unattributed += 1
            continue
        v = {m: float(r[col[m]].replace(",", "")) * UNIT.get(units[col[m]], float("nan")) for m in need}
        a = agg[rng]
        a["kernels"] += 1
        a["read"] += v["dram__bytes_read.sum"]
        a["write"] += v["dram__bytes_write.sum"]
        a["dur_s"] += v["gpu__time_duration.sum"]
    e6, ms = e6_bytes(), stage_ms()
    table = []
    for k, a in agg.items():
        f = FRAMES[k]
        meas = (a["read"] + a["write"]) / f
        emc = EMC_GBPS[k] * 1e9 * ms[k] / 1e3
        table.append({
            "stage": k, "kernels_per_frame": a["kernels"] / f,
            "measured_read_GB_per_frame": round(a["read"] / f / 1e9, 4),
            "measured_write_GB_per_frame": round(a["write"] / f / 1e9, 4),
            "measured_total_GB_per_frame": round(meas / 1e9, 4),
            "E6_analytic_GB_per_frame": round(e6[k] / 1e9, 4),
            "EMC_implied_GB_per_frame": round(emc / 1e9, 4),
            "measured_over_E6": round(meas / e6[k], 3),
            "EMC_over_measured": round(emc / meas, 3),
            "unprofiled_stage_ms": ms[k],
            "measured_GBps_at_unprofiled_time": round(meas / (ms[k] / 1e3) / 1e9, 2),
            "EMC_calibrated_GBps": EMC_GBPS[k],
            "profiled_kernel_time_ms_per_frame": round(a["dur_s"] / f * 1e3, 2)})
    with (out / "dram_bytes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    summ = {"rows": table, "unattributed_kernel_rows": unattributed,
            "reading": "measured_over_E6 ~ 1 and EMC_over_measured >> 1 => EMC busy % overstates "
                       "bytes for this access pattern; measured_over_E6 >> 1 => the stage moves more "
                       "DRAM traffic than E6's analytic count (activations / spills)."}
    (out / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "meta.json":
            files.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    gate = subprocess.run(["sudo", "-n", "nvpmodel", "-q"], capture_output=True, text=True).stdout
    meta = {"experiment": "E12b follow-up: measured DRAM bytes vs E6 bytes vs EMC-implied bytes "
                          "(analysis/E12b_review.md, owed question)",
            "power_mode": gate.splitlines()[0] if gate else "unreadable",
            "gpu_cur_freq_hz": Path("/sys/devices/platform/bus@0/17000000.gpu/devfreq/"
                                    "17000000.gpu/cur_freq").read_text().strip(),
            "config": "graphs_both, whole GPU, N=10, bf16_both rung (bench/e12b_nsys.py)",
            "ncu": "--graph-profiling node --cache-control none --replay-mode kernel; "
                   "metrics dram__bytes_read.sum, dram__bytes_write.sum, gpu__time_duration.sum",
            "caveat": "kernel replay serialises kernels, so each kernel's L2 state is the one the "
                      "previous kernel left, as in the real serial stage; concurrency inside a "
                      "stage is lost. Durations are profiled, NOT latencies -- the GB/s columns use "
                      "the unprofiled stage times.",
            "bench_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip(),
            "command": "bash bench/run_ncu_e12b_dram.sh " + str(out),
            "raw_files": files, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for t in table:
        print(json.dumps(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
