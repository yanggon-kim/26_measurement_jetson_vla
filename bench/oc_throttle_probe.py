"""Does over-current (soctherm OC) throttling slow the GPU under load? (T002 addendum)

The soctherm OC event counters (hwmon `soctherm_oc`, ocN_event_cnt, ocN_throt_en) rose during the T002 clock probe
while devfreq cur_freq stayed at the locked 1575 MHz. devfreq reports the requested clock, not a hardware slow-down,
so this measures what the GPU actually delivers: a fixed bf16 GEMM issued in 0.5 s slices; per slice the achieved
TFLOP/s (CUDA events), the OC counter deltas, devfreq cur_freq and the VIN / VDD_GPU power (sysfs).

python bench/oc_throttle_probe.py --seconds 20 --n 4096 --out results/<run>/oc_throttle_probe.csv
"""
import argparse
import csv
import glob
import os
import time

import torch


def hw(name):
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        with open(f"{h}/name") as f:
            if f.read().strip() == name:
                return h
    return None


def rd(p):
    with open(p) as f:
        return f.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--slice-s", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    oc, ina238, ina3221 = hw("soctherm_oc"), hw("ina238"), hw("ina3221")
    ocf = sorted(glob.glob(f"{oc}/oc*_event_cnt"))
    gpu_ch = next(n for n in "123" if rd(f"{ina3221}/in{n}_label") == "VDD_GPU")
    dt = getattr(torch, a.dtype)
    x = torch.randn(a.n, a.n, device="cuda", dtype=dt)
    flop = 2 * a.n ** 3
    for _ in range(20):
        x @ x
    torch.cuda.synchronize()
    t_gemm = time.perf_counter()
    for _ in range(50):
        x @ x
    torch.cuda.synchronize()
    per_slice = max(1, int(a.slice_s / ((time.perf_counter() - t_gemm) / 50)))
    rows = []
    t0 = time.perf_counter()
    prev = [int(rd(f)) for f in ocf]
    while time.perf_counter() - t0 < a.seconds:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(per_slice):
            x @ x
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e)
        cur = [int(rd(f)) for f in ocf]
        row = {"t_s": round(time.perf_counter() - t0, 3), "gemms": per_slice, "ms": round(ms, 3),
               "tflops": round(per_slice * flop / (ms * 1e-3) / 1e12, 3),
               "devfreq_hz": int(rd("/sys/class/devfreq/gpu-gpc-0/cur_freq")),
               "vin_W": round(int(rd(f"{ina238}/power1_input")) / 1e6, 2),
               "vdd_gpu_W": round(int(rd(f"{ina3221}/in{gpu_ch}_input")) * int(rd(f"{ina3221}/curr{gpu_ch}_input")) / 1e6, 2)}
        for f, c, p in zip(ocf, cur, prev):
            row[os.path.basename(f).replace("_event_cnt", "_delta")] = c - p
        prev = cur
        rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tf = [r["tflops"] for r in rows]
    print(f"slices {len(rows)}, {per_slice} GEMMs each; TFLOP/s min {min(tf)} median {sorted(tf)[len(tf)//2]} max {max(tf)}; "
          f"oc deltas total {[sum(r[k] for r in rows) for k in rows[0] if k.endswith('_delta')]}; "
          f"VIN max {max(r['vin_W'] for r in rows)} W")


if __name__ == "__main__":
    main()
