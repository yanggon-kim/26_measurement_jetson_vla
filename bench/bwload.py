"""T016 step 1 (EV-02): an SM-FREE memory-bandwidth load for the prefix bandwidth-steal sweep.

Kinds
  ce   copy engine: cudaMemcpyAsync pinned-host -> device AND device -> pinned-host on two streams
       (torch .copy_(non_blocking=True) between a pin_memory() tensor and a cuda tensor). On Orin host
       and device memory are the same LPDDR5, so every copied byte is one DRAM read + one DRAM write
       and the load's DRAM traffic = 2 x bytes copied. No kernel is launched (proved per run by an nsys
       trace of the load alone, kernel list under raw/).
  cpu  numpy copy workers pinned to cores (bench/bw_hog.py cpu_hog idea), for rates the copy engine
       cannot reach. Traffic = 2 x bytes copied as well.

It runs in its OWN PROCESS (multiprocessing, spawn), pinned to cores away from the harness: a load
thread inside the harness would contend for the GIL with the launch-heavy rung prefix -- a host-side
confound, not a bandwidth one. Shared resources with the prefix that remain, named in every meta.json:
the LPDDR5 bus and memory controller, the GPU's memory hub / SMMU path (ce), the copy engines (the
prefix itself issues no copies in its timed loop), CPU L3 / DSU (cpu kind), and the CUDA driver (ce).

Control (shared memory, set by the harness):
  rate_gbps   target DRAM traffic (read + write) in GB/s; 0 = off; <0 = as fast as possible
  window      0 = always on; 1 = on only for `win_ms` from `t0` (frame-aligned duty), where the harness
              writes t0 (time.perf_counter of the frame start) into `t0`
  bytes       cumulative bytes COPIED (x2 = DRAM traffic), for the load's own rate
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import time

CHUNK_MB = 32


def _ce_worker(ctl, cores):
    os.sched_setaffinity(0, cores)
    import torch                                                       # noqa: PLC0415
    n = CHUNK_MB * 2**20
    h1 = torch.empty(n, dtype=torch.uint8).pin_memory()
    h2 = torch.empty(n, dtype=torch.uint8).pin_memory()
    d1 = torch.empty(n, dtype=torch.uint8, device="cuda")
    d2 = torch.empty(n, dtype=torch.uint8, device="cuda")
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    ctl["ready"].value = 1
    tnext = time.perf_counter()
    while not ctl["quit"].value:
        rate = ctl["rate"].value
        if rate == 0 or (ctl["window"].value and not _in_window(ctl)):
            time.sleep(0.0005)
            tnext = time.perf_counter()
            continue
        with torch.cuda.stream(s1):
            d1.copy_(h1, non_blocking=True)
        with torch.cuda.stream(s2):
            h2.copy_(d2, non_blocking=True)
        s1.synchronize(); s2.synchronize()
        with ctl["bytes"].get_lock():
            ctl["bytes"].value += 2 * n                     # bytes copied (both directions)
        if rate > 0:                                        # pace: DRAM traffic = 2 x copied
            tnext += (2 * n * 2) / (rate * 1e9)
            dt = tnext - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                tnext = time.perf_counter()


def _cpu_worker(ctl, core, share):
    os.sched_setaffinity(0, [core])
    import numpy as np                                                 # noqa: PLC0415
    n = 64 * 2**20
    a = np.ones(n, dtype=np.uint8)
    b = np.empty_like(a)
    c = 16 * 2**20
    with ctl["ready"].get_lock():
        ctl["ready"].value += 1
    tnext = time.perf_counter()
    i = 0
    while not ctl["quit"].value:
        rate = ctl["rate"].value
        if rate == 0 or (ctl["window"].value and not _in_window(ctl)):
            time.sleep(0.0005)
            tnext = time.perf_counter()
            continue
        o = (i * c) % (n - c)
        np.copyto(b[o:o + c], a[o:o + c])
        i += 1
        with ctl["bytes"].get_lock():
            ctl["bytes"].value += c
        if rate > 0:
            tnext += (c * 2) / (rate * share * 1e9)
            dt = tnext - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                tnext = time.perf_counter()


def _in_window(ctl):
    now = time.perf_counter() - ctl["t0"].value
    return ctl["w_start"].value <= now * 1e3 < ctl["w_start"].value + ctl["win_ms"].value


class Load:
    """Start once; set .rate / .window per point; read .copied_bytes()."""

    def __init__(self, kind="ce", cores=(8, 9, 10, 11), cpu_workers=4):
        ctx = mp.get_context("spawn")
        self.kind = kind
        self.ctl = {"rate": ctx.Value(ctypes.c_double, 0.0, lock=False),
                    "window": ctx.Value(ctypes.c_int, 0, lock=False),
                    "t0": ctx.Value(ctypes.c_double, 0.0, lock=False),
                    "w_start": ctx.Value(ctypes.c_double, 0.0, lock=False),
                    "win_ms": ctx.Value(ctypes.c_double, 0.0, lock=False),
                    "bytes": ctx.Value(ctypes.c_double, 0.0),
                    "quit": ctx.Value(ctypes.c_int, 0, lock=False),
                    "ready": ctx.Value(ctypes.c_int, 0)}
        if kind == "ce":
            self.procs = [ctx.Process(target=_ce_worker, args=(self.ctl, list(cores)), daemon=True)]
            want = 1
        else:
            k = min(cpu_workers, len(cores))
            self.procs = [ctx.Process(target=_cpu_worker, args=(self.ctl, cores[i], 1.0 / k),
                                      daemon=True) for i in range(k)]
            want = k
        for p in self.procs:
            p.start()
        t = time.time()
        while self.ctl["ready"].value < want:
            if time.time() - t > 120:
                raise RuntimeError("load workers did not start")
            time.sleep(0.05)
        self.pids = [p.pid for p in self.procs]

    def set(self, rate_gbps, window=0, w_start_ms=0.0, win_ms=0.0):
        self.ctl["window"].value = window
        self.ctl["w_start"].value = w_start_ms
        self.ctl["win_ms"].value = win_ms
        self.ctl["rate"].value = rate_gbps

    def frame_start(self):
        self.ctl["t0"].value = time.perf_counter()

    def copied_bytes(self) -> float:
        return self.ctl["bytes"].value

    def close(self):
        self.ctl["quit"].value = 1
        for p in self.procs:
            p.join(timeout=10)


class Multi:
    """Several loads driven as one: the target rate is split evenly; bytes are summed."""

    def __init__(self, loads):
        self.loads = loads
        self.pids = [p for L in loads for p in L.pids]

    def set(self, rate_gbps, window=0, w_start_ms=0.0, win_ms=0.0):
        for L in self.loads:
            L.set(rate_gbps if rate_gbps < 0 else rate_gbps / len(self.loads), window, w_start_ms, win_ms)

    def frame_start(self):
        for L in self.loads:
            L.frame_start()

    def copied_bytes(self):
        return sum(L.copied_bytes() for L in self.loads)

    def close(self):
        for L in self.loads:
            L.close()


if __name__ == "__main__":
    # standalone: run the load alone at a rate for N seconds (for the nsys no-kernel proof)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="ce", choices=("ce", "cpu"))
    ap.add_argument("--rate", type=float, default=-1)
    ap.add_argument("--seconds", type=float, default=5)
    a = ap.parse_args()
    L = Load(a.kind)
    b0 = L.copied_bytes(); L.set(a.rate); t0 = time.perf_counter()
    time.sleep(a.seconds)
    L.set(0); dt = time.perf_counter() - t0
    print(f"{a.kind}: copied {(L.copied_bytes() - b0) / dt / 1e9:.2f} GB/s -> DRAM traffic "
          f"{2 * (L.copied_bytes() - b0) / dt / 1e9:.2f} GB/s")
    L.close()
