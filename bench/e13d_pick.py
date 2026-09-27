"""Pick the GEMM shapes that carry most of the prefix's excess reads (T016 1d), from kernel_bytes.csv.
Prints "M,N,K;M,N,K;..." (top --n by summed excess over frame 0) for bench/e13d_gemm_sweep.py, and writes
<out>/gemm_excess_by_shape.csv (every shape, its launches, fills, unique input bytes, excess)."""
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
R = [r for r in csv.DictReader(open(out / "kernel_bytes.csv")) if r["M"] and r["frame"] == "prefix_f0"]
agg = {}
for r in R:
    if r["gemm_kind"] != "B_is_weight":
        continue
    k = (int(r["M"]), int(r["N"]), int(r["K"]))
    a = agg.setdefault(k, {"launches": 0, "fills_B": 0, "unique_input_B": 0, "excess_B": 0, "kernels": set()})
    a["launches"] += 1; a["fills_B"] += int(r["fills_B"])
    a["unique_input_B"] += int(r["weight_B"]) + int(r["activation_B"]); a["excess_B"] += int(r["excess_read_B"])
    a["kernels"].add(r["kernel"][:70])
rows = [{"M": k[0], "N": k[1], "K": k[2], **{x: y for x, y in v.items() if x != "kernels"},
         "kernels": " | ".join(sorted(v["kernels"]))} for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["excess_B"])]
with (out / "gemm_excess_by_shape.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(";".join(f"{r['M']},{r['N']},{r['K']}" for r in rows[:n]))
