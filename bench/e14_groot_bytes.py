"""T016 step 3 (A4): GR00T N1.7 DRAM bytes per frame per variant from bench/run_ncu_e14_groot.sh chunks.

Input: <out>/ncu/e14g_<variant>_s<skip>_raw.csv (ncu --page raw, one file per chunk of <= 450 workloads of ONE
inference inside NVTX "frame"; a CUDA-graph replay is one workload). Each chunk's own units row is used (ncu
picks units per export). Per workload the innermost of the wrap ranges (vision / vlm_prefix / ae_total) is taken
from the NVTX column; the rest of "frame" (lm_head, glue, projector) is "other".
Writes <out>/dram_bytes.csv (variant x part) and <out>/summary.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

UNIT = {"sector": 32, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "": 1.0,
        "nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "ns": 1e-3, "us": 1.0, "ms": 1e3}
PARTS = ("vision", "vlm_prefix", "ae_total")


def main() -> int:
    out = Path(sys.argv[1])
    files = sorted((out / "ncu").glob("e14g_*_s*_raw.csv"),
                   key=lambda q: (re.search(r"e14g_(.+)_s(\d+)_raw", q.name).group(1),
                                  int(re.search(r"_s(\d+)_raw", q.name).group(1))))
    acc, chunks = {}, {}
    for q in files:
        v = re.search(r"e14g_(.+)_s\d+_raw", q.name).group(1)
        rr = list(csv.reader(open(q)))
        h, u, data = rr[0], rr[1], rr[2:]
        c = {x: i for i, x in enumerate(h)}
        nv = [i for i, x in enumerate(h) if "Push/Pop_Range" in x or x.startswith("NVTX")]

        def val(r, m):
            x = r[c[m]].replace(",", "")
            if x in ("", "n/a"):
                return 0.0
            if u[c[m]] not in UNIT:
                raise SystemExit(f"{q.name}: unknown unit {u[c[m]]!r} for {m}")
            return float(x) * UNIT[u[c[m]]]

        chunks.setdefault(v, []).append((q.name, len(data)))
        for r in data:
            rng = " ".join(r[i] for i in nv)
            part = next((p for p in PARTS if re.search(rf"\b{p}\b", rng)), "other")
            a = acc.setdefault(v, {}).setdefault(part, {"workloads": 0, "fills_B": 0.0, "writes_B": 0.0,
                                                       "read_req_B": 0.0, "duration_us": 0.0})
            a["workloads"] += 1
            a["fills_B"] += val(r, "lts__d_sectors_fill_device.sum") + val(r, "lts__d_sectors_fill_sysmem.sum")
            a["writes_B"] += val(r, "lts__t_sectors_op_write.sum")
            a["read_req_B"] += val(r, "lts__t_sectors_op_read.sum")
            a["duration_us"] += val(r, "gpu__time_duration.sum")
    rows = []
    for v, P in acc.items():
        tot = {k: sum(p[k] for p in P.values()) for k in ("workloads", "fills_B", "writes_B", "read_req_B", "duration_us")}
        for part, p in list(P.items()) + [("frame_total", tot)]:
            rows.append({"variant": v, "part": part, "workloads": p["workloads"],
                         "reads_GB": round(p["fills_B"] / 1e9, 4), "writes_upper_GB": round(p["writes_B"] / 1e9, 4),
                         "read_req_GB": round(p["read_req_B"] / 1e9, 4), "kernel_ms": round(p["duration_us"] / 1e3, 3)})
    with (out / "dram_bytes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    ref = {r["part"]: r for r in rows if r["variant"] == "ref"}
    summ = {"chunks": chunks, "rows": rows,
            "vs_ref_frame_reads": {v: round(r["reads_GB"] / ref["frame_total"]["reads_GB"], 4)
                                   for r in rows if r["part"] == "frame_total" for v in [r["variant"]]}
            if "frame_total" in ref else {},
            "note": "one inference per variant; reads = L2 fills from memory (DRAM reads), writes = L2 write "
                    "sectors (upper bound on DRAM writes); a CUDA-graph replay is one workload"}
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    for r in rows:
        print(r)
    print("frame reads vs ref:", summ["vs_ref_frame_reads"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
