"""T005 — does sustained load move the clock, the temperature or the throughput?

E1 reports `throttled: false`. That claim needs an experiment, not an assertion, and an
earlier version of this report cited one that produced no committed file. This is that
experiment: hold a load for as long as asked and record clock, die temperature and achieved
throughput for every repeat.

    python bench/sustained_load_probe.py --out results/<run>/raw/sustained_load.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"
THERMAL = Path("/sys/class/thermal")


def sample() -> dict:
    out = {"gpu_hz": int(Path(GPU_FREQ).read_text().strip())}
    for z in sorted(THERMAL.glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = int((z / "temp").read_text().strip()) / 1e3
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--n", type=int, default=8192)
    args = ap.parse_args()

    n = args.n
    a = torch.empty(n, n, device="cuda", dtype=torch.bfloat16).normal_()
    b = torch.empty(n, n, device="cuda", dtype=torch.bfloat16).normal_()
    rows = []
    t_start = time.perf_counter()
    for rep in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(args.iters):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        st = sample()
        row = {"repeat": rep, "elapsed_s": round(time.perf_counter() - t_start, 2),
               "ms_per_gemm": round(dt * 1e3, 4),
               "TFLOPps": round(2 * n ** 3 / dt / 1e12, 3),
               "gpu_hz": st["gpu_hz"],
               "gpu_C": st.get("gpu-thermal"), "tj_C": st.get("tj-thermal"),
               "cpu_C": st.get("cpu-thermal")}
        rows.append(row)
        print(f"  rep{rep:2d} t={row['elapsed_s']:6.1f}s  {row['ms_per_gemm']:7.2f} ms  "
              f"{row['TFLOPps']:6.2f} TFLOP/s  {row['gpu_hz']/1e6:.1f} MHz  "
              f"gpu={row['gpu_C']} C tj={row['tj_C']} C", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hz = {r["gpu_hz"] for r in rows}
    tf = [r["TFLOPps"] for r in rows]
    print(f"\n  clocks observed : {sorted(hz)} Hz -> "
          f"{'STABLE' if len(hz) == 1 else 'MOVED — THROTTLED'}")
    print(f"  throughput      : first {tf[0]:.2f}, last {tf[-1]:.2f}, "
          f"min {min(tf):.2f}, max {max(tf):.2f} TFLOP/s")
    print(f"  peak temperature: gpu {max(r['gpu_C'] for r in rows)} C, "
          f"tj {max(r['tj_C'] for r in rows)} C")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
