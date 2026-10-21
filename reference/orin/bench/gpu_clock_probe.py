"""What clock does the GPU actually run at under load, in the current power mode?

Protocol §1 says to record the device state for every run, and §3.2 says latency never
comes from ncu. The reason is here: ncu's `--clock-control base` pins the SM clock (on this
Orin, to 305.70 MHz), which is not the clock the workload sees. Run this next to any capture
so the capture's clock is recorded rather than assumed.

    python bench/gpu_clock_probe.py [--seconds 6] [--hz 50]
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time

import torch
import torch.nn.functional as F

CUR_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"


def read_freq() -> int:
    with open(CUR_FREQ) as f:
        return int(f.read().strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--S", type=int, default=51)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=4096)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(args.S, args.K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(args.N, args.K, device="cuda", dtype=torch.bfloat16)

    samples: list[int] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            samples.append(read_freq())
            time.sleep(1.0 / args.hz)

    print(f"idle cur_freq: {read_freq()}")
    t = threading.Thread(target=sampler)
    t.start()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        for _ in range(200):
            y = F.linear(x, w)
        torch.cuda.synchronize()
    stop.set()
    t.join()

    print(f"under load: n={len(samples)} min={min(samples)} "
          f"median={int(statistics.median(samples))} max={max(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
