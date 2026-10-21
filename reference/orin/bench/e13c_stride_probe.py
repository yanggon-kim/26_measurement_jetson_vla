"""T016 step 1c (EV-02 feasibility), read-only probe: stride vs bandwidth over a large buffer.

Reads ONE 64 B line every `stride` bytes across a 1 GiB device buffer (x[::stride/4] summed, which the
GPU turns into scattered 32/64 B sector reads), for strides 64 B .. 1 MiB. With channel interleaving at
granularity g over C channels, a stride that is a multiple of g*C (and not hashed away) sends every
access to one channel and the achieved line rate collapses to ~1/C of the spread case; an address hash
(XOR of higher bits into the channel bits) keeps it flat. Output: per stride, lines/s, and 'effective
GB/s' = lines x 64 B / s. The GPU has to issue kernels for this; it changes no configuration.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # FIXED line count per stride (4 Mi lines), addresses k*stride modulo an 8 GiB buffer, so every
    # stride does the same parallel work (the first version read buf/stride lines and became
    # launch/latency bound, ~0.6 ms flat, from 2 KiB up -- kept as stride_probe_v1.csv).
    n = 2**33 // 4
    x = torch.ones(n, dtype=torch.int32, device="cuda")
    rows = []
    stride = 64
    while stride <= 2**20:
        idx = (torch.arange(0, 2**22, device="cuda", dtype=torch.int64) * (stride // 4)) % n
        for _ in range(3):
            x.index_select(0, idx).sum()
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            x.index_select(0, idx).sum()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3)
        t = statistics.median(ts)
        lines = idx.numel()
        rows.append({"stride_B": stride, "lines": lines, "median_s": round(t, 7),
                     "lines_per_s": round(lines / t), "eff_GBps_64B": round(lines * 64 / t / 1e9, 3)})
        print(rows[-1], flush=True)
        stride *= 2
    with (out / "stride_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "stride_probe_meta.json").write_text(json.dumps({
        "buffer_B": 2**33, "lines_per_stride": 2**22, "method": "index_select of one int32 per stride + sum, CUDA events, median of 20",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
