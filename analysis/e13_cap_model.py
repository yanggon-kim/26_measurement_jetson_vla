#!/usr/bin/env python3
"""Bandwidth-cap model of the compiled pi0 prefix under a hard channel partition (derived, CPU only).

Input: the device's per-kernel DRAM traffic of the rung prefix,
results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/kernel_bytes.csv (ncu node-level, 2 frames).
For a GPU left with `c` of 16 channels, each kernel's time becomes max(t_i, (fills_i + writes_i) / BW_c),
BW_c = c/16 * BW_full. This assumes no extra latency from fewer channels and that each kernel spreads its
bytes evenly over its own duration, so it is a LOWER bound on the partition slowdown; E13a's contention sigma
is the measured comparison point. Every output number is [derived].

Usage: python3 analysis/e13_cap_model.py [--bw 178.5]
"""
import argparse
import csv
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/kernel_bytes.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bw", type=float, default=178.5, help="full-bus achievable GB/s (E12b M4 hog)")
    a = ap.parse_args()
    frames = defaultdict(list)
    with SRC.open() as f:
        for r in csv.DictReader(f):
            frames[r["frame"]].append((float(r["duration_us"]) * 1e-6,
                                       float(r["fills_B"]) + float(r["writes_B"])))
    print(f"source {SRC.relative_to(ROOT)}; full bus {a.bw} GB/s")
    for fr, ks in sorted(frames.items()):
        t0 = sum(t for t, _ in ks)
        b0 = sum(b for _, b in ks)
        print(f"\n{fr}: {len(ks)} kernels, {t0*1e3:.1f} ms, {b0/1e9:.2f} GB, mean {b0/t0/1e9:.1f} GB/s")
        # time share of kernels by demand
        for thr in (90, 110, 130, 150):
            share = sum(t for t, b in ks if b / t / 1e9 > thr) / t0
            print(f"  time share of kernels drawing > {thr} GB/s: {share:.3f}")
        print("  channels left | BW left GB/s | sigma_cap (lower bound)")
        for c in (16, 12, 11, 10, 9, 8):
            bw = c / 16 * a.bw * 1e9
            t1 = sum(max(t, b / bw) for t, b in ks)
            print(f"  {c:>13} | {bw/1e9:12.1f} | {t1/t0:.3f}")


if __name__ == "__main__":
    main()
