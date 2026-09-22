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


def duty_loop(do_chunk, bytes_per_chunk, duty, seconds, log, label, unit="GBps",
              extra_emit=None):
    """Run `do_chunk` for `duty` % of each 20 ms period and report the rate achieved.

    **The unit of work has to be much smaller than the window.** A whole 256 MiB copy is
    ~3.4 ms on the GPU and ~10 ms on the CPU, so at duty 25 (a 5 ms window) the inner loop
    checks its condition, does one indivisible copy, and overshoots by 2x; at duty 75 it
    overshoots past the end of the period and the sleep never runs. The first version of
    this file had that bug on top of a worse one (synchronising outside the window), and
    every duty cycle collapsed to 100 %. `do_chunk` therefore moves a slice, not the buffer.

    The requested duty is an input; the **achieved** duty is a measurement, and it is what
    the analysis must use. It is emitted with every rate sample.
    """
    on = PERIOD_S * duty / 100.0
    t_end = time.perf_counter() + seconds
    win_b, win_busy, win_t0 = 0, 0.0, time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < on:
            do_chunk()
            win_b += bytes_per_chunk
        busy = time.perf_counter() - t0
        win_busy += busy
        rest = PERIOD_S - busy
        if rest > 0:
            time.sleep(rest)
        if time.perf_counter() - win_t0 >= 1.0:
            dt = time.perf_counter() - win_t0
            emit(log, label, win_b / dt / 1e9, unit=unit,
                 achieved_duty_pct=round(100 * win_busy / dt, 2))
            if extra_emit:
                extra_emit(log, label, dt, win_b, win_busy)
            win_b, win_busy, win_t0 = 0, 0.0, time.perf_counter()


def gpu_hog(duty: float, seconds: float, log, label: str):
    import torch
    n = BUF_MIB * 1024 * 1024 // 2
    src = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    dst = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    # ~32 MiB per chunk: about 0.4 ms of GPU time, 1/12 of the shortest duty window.
    chunk = n // 8
    moved = 2 * chunk * 2                               # read + write, as E1's clone() row
    pos = [0]

    def do_chunk():
        i = pos[0]
        dst[i:i + chunk].copy_(src[i:i + chunk])
        torch.cuda.synchronize()        # bound the in-flight queue to one chunk
        pos[0] = 0 if i + chunk >= n else i + chunk

    duty_loop(do_chunk, moved, duty, seconds, log, label)


def gpu_spin_control(duty: float, seconds: float, log, label: str):
    """Same duty cycle and the same GPU time slices, but small GEMMs that fit in cache —
    bounds how much of the slowdown is time-slicing rather than lost bandwidth."""
    import torch
    a = torch.empty(512, 512, device="cuda", dtype=torch.bfloat16).normal_()
    b = torch.empty(512, 512, device="cuda", dtype=torch.bfloat16).normal_()
    n = [0]

    def do_chunk():
        a @ b
        torch.cuda.synchronize()        # same queue bound as gpu_hog
        n[0] += 1

    def also_flops(log_, label_, dt, win_b, win_busy):
        # Report BOTH: the compute rate, and the DRAM traffic this control actually moves.
        # The second is the number the E11e analysis divides by; publishing only TFLOP/s
        # (and having the caller store it in a GB/s column) is how "5.5 GB/s" got into the
        # report when the value was 5.5 TFLOP/s.
        emit(log_, label_, 2 * 512 ** 3 * n[0] / dt / 1e12, unit="TFLOPps",
             achieved_duty_pct=round(100 * win_busy / dt, 2))
        n[0] = 0

    # three matrices touched per GEMM, bf16
    duty_loop(do_chunk, 3 * 512 * 512 * 2, duty, seconds, log, label,
              extra_emit=also_flops)


def _cpu_worker(duty, seconds, cores, q, arithmetic_only):
    import numpy as np
    if cores:
        os.sched_setaffinity(0, cores)
    n = BUF_MIB * 1024 * 1024 // 8
    if arithmetic_only:
        a = np.ones(1 << 16, dtype=np.float64)          # fits in cache
        chunk = a.size
        moved = a.nbytes * 2
    else:
        src = np.ones(n, dtype=np.float64)
        dst = np.empty(n, dtype=np.float64)
        # ~16 MiB per chunk: about 0.6 ms, well inside the shortest duty window. A whole
        # 256 MiB np.copyto is ~10 ms, half the period, and cannot be interrupted.
        chunk = n // 16
        moved = 2 * chunk * 8
    pos = [0]

    def do_chunk():
        if arithmetic_only:
            # np.multiply(..., out=a), not `a *= …`: augmented assignment REBINDS the name,
            # which makes `a` local to this closure and raises UnboundLocalError. Every
            # cpu_spin_control worker died on its first iteration and the control produced
            # an empty log — a blank row where §13.5 requires a measurement.
            np.multiply(a, 1.0000001, out=a)
        else:
            i = pos[0]
            np.copyto(dst[i:i + chunk], src[i:i + chunk])
            pos[0] = 0 if i + chunk >= n else i + chunk

    on = PERIOD_S * duty / 100.0
    t_end = time.perf_counter() + seconds
    win_b, win_busy, win_t0 = 0, 0.0, time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < on:
            do_chunk()
            win_b += moved
        busy = time.perf_counter() - t0
        win_busy += busy
        rest = PERIOD_S - busy
        if rest > 0:
            time.sleep(rest)
        if time.perf_counter() - win_t0 >= 1.0:
            dt = time.perf_counter() - win_t0
            q.put((win_b / dt / 1e9, 100 * win_busy / dt))
            win_b, win_busy, win_t0 = 0, 0.0, time.perf_counter()


def cpu_hog(duty, seconds, log, label, workers, cores, arithmetic_only=False):
    """Sum the workers' rates on a fixed 1 s cadence.

    The queue is drained completely each pass and the latest report from each worker is
    kept. Summing whatever happened to arrive inside a 1 s window instead — the first
    version — counted a variable number of workers per sample, which is why the rate a
    cpu_hog reported "alone" fell as its duty cycle rose.
    """
    q = mp.Queue()
    ps = [mp.Process(target=_cpu_worker,
                     args=(duty, seconds, cores, q, arithmetic_only))
          for _ in range(workers)]
    for p in ps:
        p.start()
    t_end = time.perf_counter() + seconds
    latest: dict[int, tuple[float, float]] = {}
    seq, last = 0, time.perf_counter()
    while time.perf_counter() < t_end:
        try:
            latest[seq % workers] = q.get(timeout=0.2)
            seq += 1
        except Exception:                                          # noqa: BLE001
            pass
        if time.perf_counter() - last >= 1.0 and latest:
            emit(log, label, sum(v[0] for v in latest.values()),
                 achieved_duty_pct=round(
                     sum(v[1] for v in latest.values()) / len(latest), 2))
            last = time.perf_counter()
    for p in ps:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


def emit(log, label, rate, unit="GBps", **extra):
    line = json.dumps({"t": round(time.time(), 3), "label": label,
                       "rate": round(rate, 3), "unit": unit, **extra})
    print(line, flush=True)
    if log:
        with open(log, "a") as f:
            f.write(line + "\n")


def self_test(cores, workers) -> int:
    """Run every hog kind for two seconds and fail if one produced no samples.

    A hog that dies in a worker process prints its traceback to a log nobody reads and
    then reports nothing; mempressure stores a blank and carries on for 47 minutes. That
    is how cpu_spin_control reached a finished result directory as an empty row. Two
    seconds per kind up front is cheaper than finding out afterwards.
    """
    import tempfile
    bad = []
    for kind in ("gpu", "gpu_spin_control", "cpu", "cpu_spin_control"):
        with tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False) as f:
            log = f.name
        if kind == "gpu":
            gpu_hog(100.0, 2.0, log, kind)
        elif kind == "gpu_spin_control":
            gpu_spin_control(100.0, 2.0, log, kind)
        else:
            cpu_hog(100.0, 2.0, log, kind, workers, cores,
                    arithmetic_only=kind.endswith("spin_control"))
        lines = [ln for ln in open(log).read().splitlines() if ln.strip().startswith("{")]
        rates = [json.loads(ln) for ln in lines]
        gb = [r for r in rates if r.get("unit") == "GBps"]
        os.unlink(log)
        ok = bool(gb) and all(r.get("achieved_duty_pct") is not None for r in gb)
        print(f"  {kind:18s} {'ok  ' if ok else 'FAIL'} "
              f"{len(gb)} GBps sample(s)"
              + (f", median {sorted(r['rate'] for r in gb)[len(gb)//2]:.1f} GB/s" if gb else ""),
              flush=True)
        if not ok:
            bad.append(kind)
    if bad:
        print(f"bw_hog self-test FAILED for: {', '.join(bad)}")
        return 1
    print("bw_hog self-test passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("gpu", "cpu", "gpu_spin_control", "cpu_spin_control"))
    ap.add_argument("--duty", type=float, default=100.0)
    ap.add_argument("--self-test", action="store_true",
                    help="run every kind for 2 s and fail if one produced no samples")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cores", default="", help="comma-separated core ids for the CPU hog")
    args = ap.parse_args()
    cores = {int(c) for c in args.cores.split(",") if c.strip()}
    if args.self_test:
        return self_test(cores, args.workers)
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
