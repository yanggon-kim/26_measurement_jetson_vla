"""T003 E2 post-processing: the DRAM floor of each expert projection with WEIGHT bytes only.

kernels.csv's floor_ms counts weights + input + output activations as DRAM bytes. Back to back, the activations stay
in the 32 MiB L2 (the same x feeds every call, the output buffer is reused), so only the weights must come from DRAM;
counting the activations put several S = 256 rows below their floor (kernel_over_floor < 1), which no kernel can
really do. The E1 follow-up (results/20261022_E1_bw_gemm_stream) shows the same calls stream weights at 176-293 GB/s
of "weight + activation" bytes but ≤ 216 GB/s of weight bytes, below E1's 258 GB/s ceiling.

Writes kernels_floors.csv: per row the weight-only DRAM floor at the measured best bandwidth, the compute floor at the
measured bf16 ceiling, the floor (max), kernel_cold_ms ÷ floor, and the achieved weight-streaming GB/s.

python bench/e2_floors.py --e2 results/<E2> --hw results/<E1>/hw_constants.json
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", required=True)
    ap.add_argument("--hw", required=True)
    a = ap.parse_args()
    hw = json.load(open(a.hw))
    bw = hw["measured_bw_GBps"]["best"]["value"] * 1e9
    fl = hw["measured_tflops"]["bf16"]["value"] * 1e12
    out = []
    for r in csv.DictReader(open(os.path.join(a.e2, "kernels.csv"))):
        wb, flops = int(r["weight_bytes"]), int(r["flops"])
        k = float(r["kernel_cold_ms"])
        mem = wb / bw * 1e3
        cmp_ = flops / fl * 1e3
        floor = max(mem, cmp_)
        out.append({"op": r["op"], "S": r["S"], "K": r["K"], "N": r["N"], "weight_bytes": wb, "flops": flops,
                    "kernel_cold_ms": k, "kernel_hot_ms": r["kernel_hot_ms"], "issue_ms": r["issue_ms"],
                    "floor_weight_dram_ms": round(mem, 6), "floor_compute_ms": round(cmp_, 6), "floor_ms": round(floor, 6),
                    "floor_bound": "memory" if mem >= cmp_ else "compute", "kernel_over_floor": round(k / floor, 3),
                    "weight_GBps_achieved": round(wb / (k / 1e3) / 1e9, 2),
                    "pct_of_measured_peak_bw": round(100 * wb / (k / 1e3) / bw, 1),
                    "peaks_source": a.hw})
    with open(os.path.join(a.e2, "kernels_floors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r["S"] in ("1", "51", "256"):
            print(r["op"], r["S"], "cold", round(r["kernel_cold_ms"] * 1e3, 1), "us; x floor", r["kernel_over_floor"],
                  "; weight GB/s", r["weight_GBps_achieved"], f"({r['pct_of_measured_peak_bw']} %)")


if __name__ == "__main__":
    main()
