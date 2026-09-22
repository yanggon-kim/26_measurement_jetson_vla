"""T010 / E11e — a background memory-bandwidth load, as a separate process (protocol §13.5).

The hybrid model assumes the GPU-side prefix runs at full speed while the memory system is busy
denoising (`prefix_slowdown = 1.0`, **[ASSUMPTION]**). There is no PIM to test that against, so
§13.5 specifies a software proxy and is explicit about what it is not:

  "A real PIM occupies DRAM channels through the command bus and blocks ordinary accesses to
   them; a software hog instead competes at the memory controller and in the caches, and
   additionally takes GPU time slices (`gpu_hog`) or CPU cores (`cpu_hog`). The controls bound
   those side effects; the result is the sensitivity of the GPU work to lost memory bandwidth,
   **not a measurement of PIM interference**."

Duty cycle d over a 20 ms period: copy for d*20 ms, sleep the rest. The achieved rate is
printed once a second so the foreground run can report the background rate *alone* and
*during*. GB/s counts read + write, the same convention as E1's `clone()` row
(`bench/micro_peak.py`).

    python bench/bw_hog.py --kind gpu --duty 50 --seconds 90 --log raw/bw_hog_gpu_50.log
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

PERIOD_S = 0.020
BUF_MIB = 256


def gpu_hog(duty: float, seconds: float, log, label: str):
    import torch
    n = BUF_MIB * 1024 * 1024 // 2
    src = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    dst = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    moved = 2 * n * 2                                   # read + write, as E1's clone() row
    on = PERIOD_S * duty / 100.0
    t_end = time.perf_counter() + seconds
    win_b, win_t0 = 0, time.perf_counter()
    # One copy takes ~3.4 ms of GPU time. Queuing copies for `on` seconds and only THEN
    # synchronising drains a queue far longer than the 20 ms period, so `rest` was always
    # negative, the sleep never ran, and every duty cycle collapsed to 100 % -- which is why
    # the first run showed "duty cycle barely matters". Synchronise inside the window so the
    # GPU is actually idle for the rest of the period.
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < on:
            dst.copy_(src)
            torch.cuda.synchronize()          # bound the in-flight queue to one copy
            win_b += moved
        rest = PERIOD_S - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        if time.perf_counter() - win_t0 >= 1.0:
            dt = time.perf_counter() - win_t0
            emit(log, label, win_b / dt / 1e9)
            win_b, win_t0 = 0, time.perf_counter()


def gpu_spin_control(duty: float, seconds: float, log, label: str):
    """Same duty cycle and the same GPU time slices, but small GEMMs that fit in cache —
    bounds how much of the slowdown is time-slicing rather than lost bandwidth."""
    import torch
    a = torch.empty(512, 512, device="cuda", dtype=torch.bfloat16).normal_()
    b = torch.empty(512, 512, device="cuda", dtype=torch.bfloat16).normal_()
    on = PERIOD_S * duty / 100.0
    t_end = time.perf_counter() + seconds
    win_t0, win_n = time.perf_counter(), 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < on:
            a @ b
            torch.cuda.synchronize()          # same queue bound as gpu_hog
            win_n += 1
        rest = PERIOD_S - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        if time.perf_counter() - win_t0 >= 1.0:
            dt = time.perf_counter() - win_t0
            # Report BOTH: the compute rate, and the DRAM traffic this control actually
            # moves. The second is the number the E11e analysis divides by; publishing only
            # TFLOP/s (and having the caller store it in a GB/s column) is how "5.5 GB/s"
            # got into the report when the value was 5.5 TFLOP/s.
            emit(log, label, 2 * 512 ** 3 * win_n / dt / 1e12, unit="TFLOPps")
            emit(log, label, 3 * 512 * 512 * 2 * win_n / dt / 1e9, unit="GBps")
            win_t0, win_n = time.perf_counter(), 0


def _cpu_worker(duty, seconds, cores, q, arithmetic_only):
    import numpy as np
    if cores:
        os.sched_setaffinity(0, cores)
    n = BUF_MIB * 1024 * 1024 // 8
    if arithmetic_only:
        a = np.ones(1 << 16, dtype=np.float64)          # fits in cache
    else:
        src = np.ones(n, dtype=np.float64)
        dst = np.empty(n, dtype=np.float64)
    moved = 2 * n * 8
    on = PERIOD_S * duty / 100.0
    t_end = time.perf_counter() + seconds
    win_b, win_t0 = 0, time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < on:
            if arithmetic_only:
                a *= 1.0000001
                win_b += a.nbytes * 2
            else:
                np.copyto(dst, src)
                win_b += moved
        rest = PERIOD_S - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
        if time.perf_counter() - win_t0 >= 1.0:
            dt = time.perf_counter() - win_t0
            q.put(win_b / dt / 1e9)
            win_b, win_t0 = 0, time.perf_counter()


def cpu_hog(duty, seconds, log, label, workers, cores, arithmetic_only=False):
    q = mp.Queue()
    ps = [mp.Process(target=_cpu_worker,
                     args=(duty, seconds, cores, q, arithmetic_only))
          for _ in range(workers)]
    for p in ps:
        p.start()
    t_end = time.perf_counter() + seconds
    acc, last = 0.0, time.perf_counter()
    while time.perf_counter() < t_end:
        try:
            acc += q.get(timeout=0.5)
        except Exception:                                          # noqa: BLE001
            pass
        if time.perf_counter() - last >= 1.0:
            emit(log, label, acc)
            acc, last = 0.0, time.perf_counter()
    for p in ps:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


def emit(log, label, rate, unit="GBps"):
    line = json.dumps({"t": round(time.time(), 3), "label": label,
                       "rate": round(rate, 3), "unit": unit})
    print(line, flush=True)
    if log:
        with open(log, "a") as f:
            f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=("gpu", "cpu", "gpu_spin_control", "cpu_spin_control"))
    ap.add_argument("--duty", type=float, default=100.0)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cores", default="", help="comma-separated core ids for the CPU hog")
    args = ap.parse_args()
    cores = {int(c) for c in args.cores.split(",") if c.strip()}
    label = f"{args.kind}_{int(args.duty)}"
    if args.log:
        open(args.log, "w").close()
    if args.kind == "gpu":
        gpu_hog(args.duty, args.seconds, args.log, label)
    elif args.kind == "gpu_spin_control":
        gpu_spin_control(args.duty, args.seconds, args.log, label)
    elif args.kind == "cpu":
        cpu_hog(args.duty, args.seconds, args.log, label, args.workers, cores)
    else:
        cpu_hog(args.duty, args.seconds, args.log, label, args.workers, cores,
                arithmetic_only=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
