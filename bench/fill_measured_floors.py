"""T005 — fill the `measured` rows that every floor.csv has been carrying empty since T002.

`bench/run_e3.py` writes two rows per region: one against the 204.8 GB/s / 42 TFLOP/s spec and
one with `peak_source = measured` left blank, because nothing had measured this device's peaks.
E1 has now measured them, so the blanks can be filled. Protocol §8: a floor in this campaign is
computed from THIS device's peaks, not from a data sheet.

    python bench/fill_measured_floors.py                 # every results/*/floor.csv
    python bench/fill_measured_floors.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

E1 = REPO / "results" / "20260922_E1_peak" / "hw_constants.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--e1", default=str(E1))
    args = ap.parse_args()

    mp = json.loads(Path(args.e1).read_text())["measured_peaks"]
    bw = mp["peak_memory_bandwidth_GBps"]
    fl = mp["peak_bf16_TFLOPps"]
    src = Path(args.e1).relative_to(REPO)
    note = (f"measured peaks from {src} (T005/E1): {bw} GB/s, {fl} TFLOP/s. "
            f"Bandwidth on this device varies run to run by up to 1.19x with clocks and "
            f"temperature constant, so this floor carries that spread — see "
            f"reports/E1_E2_microbench.md.")

    changed = 0
    for f in sorted((REPO / "results").glob("*/floor.csv")):
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        hit = False
        for r in rows:
            if r.get("peak_source") != "measured":
                continue
            b = float(r["bytes_read"] or 0)
            fp = float(r["flops"] or 0)
            mem = b / (bw * 1e9) * 1e3
            cmp_ = fp / (fl * 1e12) * 1e3 if fp else 0.0
            r["peak_GBps"] = bw
            r["peak_TFLOPps"] = fl
            r["floor_mem_ms"] = round(mem, 4)
            r["floor_compute_ms"] = round(cmp_, 4)
            r["floor_ms"] = round(max(mem, cmp_), 4)
            r["bound"] = "memory" if mem >= cmp_ else "compute"
            r["note"] = note
            hit = True
            changed += 1
        if hit and not args.dry_run:
            with f.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        if hit:
            print(f"  {'(dry) ' if args.dry_run else ''}{f.relative_to(REPO)}")
            for r in rows:
                if r.get("peak_source") == "measured":
                    print(f"      {r['region']:12s} {float(r['bytes_read'])/1e6:9.2f} MB -> "
                          f"floor {r['floor_ms']} ms ({r['bound']})")
    print(f"\n{changed} measured rows filled at {bw} GB/s / {fl} TFLOP/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
