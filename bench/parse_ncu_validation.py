"""T003 step 3: validation.csv from the ncu raw-page exports of bench/ncu_validation.py.

Per case and kernel launch: measured / analytic DRAM read bytes (two candidate counters), write bytes, and FLOPs
(SIMT and tensor separately). Analytic values come from the ANALYTIC line of each case's stdout log.
Byte counters (× 32 B per sector):
  read_fill   = lts__d_sectors_fill_sysmem.sum
  read_miss   = lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum
  write       = lts__t_sectors_aperture_sysmem_op_write.sum
FLOP counters:
  simt        = fadd + fmul + 2 * ffma (+ the h* and d* variants), smsp__sass_thread_inst_executed_op_*_pred_on.sum
  tensor      = sum of sm__ops_path_tensor_src_*.sum (excluding *_realtime; they duplicate)

python bench/parse_ncu_validation.py --dir results/<run>
"""
import argparse
import csv
import glob
import json
import os
import re


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    out = []
    for raw in sorted(glob.glob(os.path.join(a.dir, "ncu", "val_*_raw.csv"))):
        case = re.search(r"val_(.+)_raw\.csv", raw).group(1)
        log = os.path.join(a.dir, "raw", f"val_{case}.stdout.log")
        ana = json.loads(next(l for l in open(log) if l.startswith("ANALYTIC ")).split(" ", 1)[1])
        rows = list(csv.reader(open(raw)))
        hdr = rows[0]
        for li, vals in enumerate(rows[2:]):
            d = dict(zip(hdr, vals))
            fill = num(d.get("lts__d_sectors_fill_sysmem.sum", 0)) * 32
            miss = num(d.get("lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum", 0)) * 32
            wr = num(d.get("lts__t_sectors_aperture_sysmem_op_write.sum", 0)) * 32
            simt = 0.0
            for p in ("f", "h", "d"):
                simt += num(d.get(f"smsp__sass_thread_inst_executed_op_{p}add_pred_on.sum", 0))
                simt += num(d.get(f"smsp__sass_thread_inst_executed_op_{p}mul_pred_on.sum", 0))
                simt += 2 * num(d.get(f"smsp__sass_thread_inst_executed_op_{p}fma_pred_on.sum", 0))
            tensor = sum(num(v) for k, v in d.items()
                         if k.startswith("sm__ops_path_tensor_src_") and k.endswith(".sum") and "realtime" not in k)
            tensor_by_src = {k: num(v) for k, v in d.items()
                             if k.startswith("sm__ops_path_tensor_src_") and k.endswith(".sum")
                             and "realtime" not in k and num(v) > 0}
            r = {"case": case, "launch": li, "kernel": d.get("Kernel Name", ""), "dtype": ana.get("dtype"),
                 "analytic_read_B": ana["read"], "analytic_write_B": ana["write"], "analytic_flops": ana["flops"],
                 "read_fill_B": int(fill), "read_miss_B": int(miss), "write_B": int(wr),
                 "ratio_read_fill": round(fill / ana["read"], 4) if ana["read"] else "",
                 "ratio_read_miss": round(miss / ana["read"], 4) if ana["read"] else "",
                 "ratio_write": round(wr / ana["write"], 4) if ana["write"] > 64 else "n/a: analytic write < 64 B",
                 "simt_flops": int(simt), "tensor_flops": int(tensor),
                 "ratio_simt_flops": round(simt / ana["flops"], 4) if ana["flops"] else "",
                 "ratio_tensor_flops": round(tensor / ana["flops"], 4) if ana["flops"] else "",
                 "tensor_counters_nonzero": json.dumps(tensor_by_src),
                 "duration_ns": d.get("gpu__time_duration.sum", ""),
                 "sm_hz": d.get("sm__cycles_elapsed.avg.per_second", ""),
                 "notes": "; ".join(x for x in (ana.get("bytes_note", ""), ana.get("flops_note", "")) if x)}
            out.append(r)
    with open(os.path.join(a.dir, "validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r["case"], r["launch"], r["kernel"][:40], "read fill/miss", r["ratio_read_fill"], r["ratio_read_miss"],
              "write", r["ratio_write"], "simt", r["ratio_simt_flops"], "tensor", r["ratio_tensor_flops"])


if __name__ == "__main__":
    main()
