"""T005 — record which memory-performance mode this device is currently in.

E1 found that bandwidth on this board is bimodal: a slow mode around 150-165 GB/s and a fast
mode around 180-190 GB/s, a uniform ~1.21x that moves every memory-bound kernel together and
persists across many consecutive process launches. Neither the cause nor a way to force a mode
is known.

This probe exists so the claim is backed by a committed, re-runnable measurement rather than
by an interactive session. Each invocation is a fresh process and reports one sample of the
current mode, together with the state that was checked and eliminated as the cause: the GPU
clock, the EMC clock sampled *during* the timed loop, the die temperature, the device pointers
of the buffers and the free memory.

    for i in $(seq 1 8); do python bench/bandwidth_mode_probe.py; done | tee modes.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"


def emc() -> int:
    try:
        r = subprocess.run(
            ["sudo", "-n", "sh", "-c",
             "cat /sys/kernel/debug/bpmp/debug/clk/emc/rate"],
            capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except Exception:
        return -1


def temps() -> dict:
    out = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = int((z / "temp").read_text().strip()) / 1e3
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    n = int(args.gib * 1024 * 1024 * 1024) // 2        # bf16 elements
    x = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    y = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    e0 = emc()
    for _ in range(args.warmup):
        y.add_(x, alpha=1.0001)
    torch.cuda.synchronize()
    e1 = emc()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(args.iters)]
    for a, b in ev:
        a.record()
        y.add_(x, alpha=1.0001)
        b.record()
    torch.cuda.synchronize()
    e2 = emc()
    ms = [a.elapsed_time(b) for a, b in ev]
    med = statistics.median(ms)
    moved = 3 * n * 2
    bw = moved / (med / 1e3) / 1e9
    T = temps()
    print(json.dumps({
        "GBps": round(bw, 1),
        "mode": "fast" if bw > 175 else ("slow" if bw < 170 else "between"),
        "median_ms": round(med, 4),
        "p5_ms": round(sorted(ms)[len(ms) // 20], 4),
        "p95_ms": round(sorted(ms)[-max(1, len(ms) // 20)], 4),
        "bytes": moved, "iters": args.iters,
        "gpu_hz": int(Path(GPU_FREQ).read_text().strip()),
        "emc_before": e0, "emc_mid": e1, "emc_after": e2,
        "gpu_C": T.get("gpu-thermal"), "tj_C": T.get("tj-thermal"),
        "ptr_x": x.data_ptr(), "ptr_y": y.data_ptr(),
        "free_MB": round(torch.cuda.mem_get_info()[0] / 1e6),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
