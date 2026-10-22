"""The GPU clock under load (ported from reference/orin/bench/gpu_clock_probe.py; T002 step 3).

Runs a bf16 GEMM loop for --seconds and samples /sys/class/devfreq/gpu-gpc-0/cur_freq at --hz from a thread.
Writes a tidy CSV (t_s, phase, cur_freq_hz) with an idle phase before and after the load.

python bench/gpu_clock_probe.py --seconds 10 --hz 20 --out results/<run>/clock_probe.csv
"""
import argparse
import csv
import statistics
import threading
import time

import torch

CUR_FREQ = "/sys/class/devfreq/gpu-gpc-0/cur_freq"


def read_freq():
    with open(CUR_FREQ) as f:
        return int(f.read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=4096, help="square bf16 GEMM size")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    x = torch.randn(a.n, a.n, device="cuda", dtype=torch.bfloat16)
    rows, phase = [], {"v": "idle_before"}
    stop = threading.Event()
    t0 = time.perf_counter()

    def sampler():
        k = 0
        while not stop.is_set():
            rows.append({"t_s": round(time.perf_counter() - t0, 4), "phase": phase["v"], "cur_freq_hz": read_freq()})
            k += 1
            time.sleep(max(0.0, t0 + k / a.hz - time.perf_counter()))

    th = threading.Thread(target=sampler)
    th.start()
    time.sleep(2.0)
    phase["v"] = "load"
    n_gemm = 0
    tl = time.perf_counter()
    while time.perf_counter() - tl < a.seconds:
        for _ in range(20):
            x @ x
        n_gemm += 20
        torch.cuda.synchronize()
    phase["v"] = "idle_after"
    time.sleep(2.0)
    stop.set()
    th.join()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "phase", "cur_freq_hz"])
        w.writeheader()
        w.writerows(rows)
    load = [r["cur_freq_hz"] for r in rows if r["phase"] == "load"]
    print({"n_load_samples": len(load), "min": min(load), "median": int(statistics.median(load)), "max": max(load),
           "distinct": sorted(set(load)), "gemms": n_gemm, "gemm_n": a.n})


if __name__ == "__main__":
    main()
