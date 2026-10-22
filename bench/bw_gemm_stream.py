"""T003 E1 addendum: DRAM bandwidth drawn by back-to-back skinny GEMMs streaming ≥ 1 GiB of distinct weights.

E2 found gate/up_proj at S = 256 taking 38.4 µs per call back to back for ≈ 11 MB, i.e. ≈ 286 GB/s, above E1's best
elementwise / reduction kernel (258 GB/s), while ncu shows each call reads exactly its weight bytes from DRAM. This
measures the sustained rate with no possible weight reuse: one pass issues F.linear(x[S, K], W_i) over n distinct
W_i whose total is ≥ 1 GiB (≥ 32× the L2), back to back on one stream, between one CUDA-event pair; GB/s = (weight +
activation bytes of the pass) ÷ pass time. Each (shape, S) is repeated; the gate runs first; OC counters around it.

python bench/bw_gemm_stream.py --proc-tag p1 --out results/<run>
"""
import argparse
import csv
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    g0 = gate.run_gate()
    if not g0["gate_pass"]:
        raise SystemExit(f"gate failed {g0['checks']}")
    rows = []
    for (N, K) in [(4096, 1024), (2048, 1024), (1024, 4096)]:
        wb = N * K * 2
        n = int(a.gib * (1 << 30)) // wb
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
        for S in (1, 51, 256):
            x = torch.randn(S, K, device="cuda", dtype=torch.bfloat16)

            def one_pass():
                for w in ws:
                    F.linear(x, w)

            for _ in range(2):
                one_pass()
            torch.cuda.synchronize()
            time.sleep(1.0)
            oc0 = gate.oc_counts()
            ms = []
            for _ in range(a.repeats):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                one_pass()
                e.record()
                e.synchronize()
                ms.append(s.elapsed_time(e))
            oc1 = gate.oc_counts()
            nb = n * (wb + 2 * S * (K + N))
            med = statistics.median(ms)
            rows.append({"proc": a.proc_tag, "N": N, "K": K, "S": S, "n_weights": n, "pass_bytes": nb,
                         "median_pass_ms": round(med, 4), "min_pass_ms": round(min(ms), 4), "max_pass_ms": round(max(ms), 4),
                         "repeats": len(ms), "GBps_median": round(nb / (med / 1e3) / 1e9, 2),
                         "GBps_best": round(nb / (min(ms) / 1e3) / 1e9, 2), "us_per_call": round(med * 1e3 / n, 3),
                         "oc_event_delta": sum(oc1[k] - oc0[k] for k in oc0)})
            print(rows[-1], flush=True)
        del ws
        torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"bw_gemm_stream_{a.proc_tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
