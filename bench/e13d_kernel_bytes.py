"""T016 step 1d (EV-02c): kernel_bytes.csv -- per-launch DRAM traffic of the pi0 rung prefix.

Inputs (in <out>): ncu/e13d_<tag>_raw.csv (per-kernel ncu, bench/run_ncu_e13d.sh) and raw/shapes_<tag>.csv
(torch.profiler op/shape map, bench/e13d_target.py --mode shapes).

Per launch: frame, launch order, kernel name, grid, block, fills (B, = L2 sector fills from memory x 32),
writes (B, L2 write sectors x 32, an upper bound on DRAM writes), read requests (B), duration, and -- when
the launch is matched to a profiler op -- the op, its input shapes, and for GEMMs M, N, K and the UNIQUE
operand bytes (weight / activation / output). excess_read_B = fills - unique input operand bytes.

Matching ncu launches to profiler kernels: the two launch sequences are aligned by kernel name
(difflib on the first 48 characters); unmatched launches keep op = "" and are counted in the report.
"""
from __future__ import annotations

import ast
import csv
import difflib
import json
import sys
from pathlib import Path

UNIT = {"sector": 32, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
        "nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "ns": 1e-3, "us": 1.0, "ms": 1e3}
GEMM_OPS = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::linear", "aten::matmul",
            "aten::_scaled_mm")


def norm(n: str) -> str:
    return n.replace("void ", "").strip()[:48]


def gemm_shape(op: str, shapes):
    try:
        s = [x for x in shapes if isinstance(x, list) and x]
        if op == "aten::mm":
            (m, k), (_, n) = s[0], s[1]
            return m, n, k, 1, "B_is_weight"
        if op == "aten::addmm":
            (m, k), (_, n) = s[1], s[2]
            return m, n, k, 1, "B_is_weight"
        if op in ("aten::bmm", "aten::baddbmm"):
            a, b = (s[0], s[1]) if op == "aten::bmm" else (s[1], s[2])
            return a[-2], b[-1], a[-1], a[0], "both_activations"
        if op == "aten::linear":
            x, w = s[0], s[1]
            m = 1
            for d in x[:-1]:
                m *= d
            return m, w[0], w[1], 1, "B_is_weight"
    except (ValueError, IndexError, TypeError):
        return None
    return None


def main() -> int:
    out = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "rung"
    el = 2                                                     # bf16 rung; fp32 kernels flagged below
    rows = list(csv.reader(open(out / "ncu" / f"e13d_{tag}_raw.csv")))
    hdr, units, data = rows[0], rows[1], rows[2:]
    c = {h: i for i, h in enumerate(hdr)}
    nv = next(i for i, h in enumerate(hdr) if "Push/Pop_Range" in h)

    def v(r, m):
        x = r[c[m]].replace(",", "")
        return float(x) * UNIT.get(units[c[m]], 1.0) if x not in ("", "n/a") else 0.0

    launches = []
    for r in data:
        rng = r[nv]
        fr = next((f"prefix_f{k}" for k in range(8) if f"prefix_f{k}" in rng), None)
        if fr is None:
            continue
        launches.append({"frame": fr, "kernel": r[c["Kernel Name"]], "grid": r[c["Grid Size"]],
                         "block": r[c["Block Size"]],
                         "fills_B": v(r, "lts__d_sectors_fill_device.sum") + v(r, "lts__d_sectors_fill_sysmem.sum"),
                         "writes_B": v(r, "lts__t_sectors_op_write.sum"),
                         "read_req_B": v(r, "lts__t_sectors_op_read.sum"),
                         "read_miss_B": v(r, "lts__t_sectors_op_read_lookup_miss.sum"),
                         "duration_us": v(r, "gpu__time_duration.sum")})
    prof = list(csv.DictReader(open(out / "raw" / f"shapes_{tag}.csv")))
    frames = sorted({x["frame"] for x in launches})
    out_rows, matched = [], 0
    for fr in frames:
        L = [x for x in launches if x["frame"] == fr]
        sm = difflib.SequenceMatcher(a=[norm(x["kernel"]) for x in L],
                                     b=[norm(p["kernel"]) for p in prof], autojunk=False)
        pmap = {}
        for blk in sm.get_matching_blocks():
            for j in range(blk.size):
                pmap[blk.a + j] = prof[blk.b + j]
        for i, x in enumerate(L):
            p = pmap.get(i)
            row = {"frame": fr, "launch": i, **{k: x[k] for k in ("kernel", "grid", "block")},
                   **{k: round(x[k]) for k in ("fills_B", "writes_B", "read_req_B", "read_miss_B")},
                   "duration_us": round(x["duration_us"], 3), "op": "", "input_shapes": "",
                   "M": "", "N": "", "K": "", "batch": "", "weight_B": "", "activation_B": "",
                   "output_B": "", "excess_read_B": "", "gemm_kind": ""}
            if p:
                matched += 1
                row["op"], row["input_shapes"] = p["cpu_op"], p["input_shapes"]
                try:
                    shapes = ast.literal_eval(p["input_shapes"])
                except (ValueError, SyntaxError):
                    shapes = []
                g = gemm_shape(p["cpu_op"], shapes) if p["cpu_op"] in GEMM_OPS else None
                if g:
                    m, n, k, b, kind = g
                    e = 4 if ("sgemm" in x["kernel"] or "f32" in x["kernel"]) else el
                    wb = (b * k * n if kind == "both_activations" else k * n) * e
                    ab = b * m * k * e
                    row.update({"M": m, "N": n, "K": k, "batch": b, "gemm_kind": kind, "weight_B": wb,
                                "activation_B": ab, "output_B": b * m * n * e,
                                "excess_read_B": round(x["fills_B"] - wb - ab)})
            out_rows.append(row)
    with (out / "kernel_bytes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    per = {}
    for fr in frames:
        R = [r for r in out_rows if r["frame"] == fr]
        g = [r for r in R if r["M"] != ""]
        per[fr] = {"launches": len(R), "fills_GB": sum(r["fills_B"] for r in R) / 1e9,
                   "writes_GB": sum(r["writes_B"] for r in R) / 1e9,
                   "read_req_GB": sum(r["read_req_B"] for r in R) / 1e9,
                   "duration_ms": sum(r["duration_us"] for r in R) / 1e3,
                   "gemm_launches": len(g), "gemm_fills_GB": sum(r["fills_B"] for r in g) / 1e9,
                   "gemm_unique_input_GB": sum(r["weight_B"] + r["activation_B"] for r in g) / 1e9,
                   "gemm_excess_GB": sum(r["excess_read_B"] for r in g) / 1e9}
    summ = {"per_frame": per, "matched_launches": matched, "total_launches": len(out_rows),
            "e12b_reference_GB": {"reads": 17.80, "writes_upper": 3.83},
            "note": "fills = DRAM reads (L2 fills from memory); writes = L2 write sectors (upper bound)"}
    (out / "kernel_bytes_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
