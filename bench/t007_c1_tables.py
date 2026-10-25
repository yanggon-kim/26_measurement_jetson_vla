"""T007 steps 2-3: C1 tables from the ncu --set full captures of one model x mode (bench/t007_c1.py), joined to the
nsys capture of the same configuration (bench/t006_capture.py, kernels.csv) on launch order + kernel base name + grid
+ block (bench/t006_roofline.join: ±8 positions; a row matching by name only is not counted in coverage).

Per kernel (ncu/kernels.csv): nsys duration (time), SM throughput %, on-chip memory throughput % (onchip_mem_pct =
gpu__compute_memory_throughput: the max of the memory units' throughputs; this chip exposes no dram__ counters (T001), so
it does NOT include DRAM), achieved occupancy %, tensor-pipe active %, the 17 stall reasons (warps per issue-active
cycle), DRAM GB/s from the validated bytes ÷ nsys duration and its % of the measured peak (hw_constants.json), and a
verdict, ncu's speed-of-light rule with memory = max(on-chip %, DRAM % of measured peak): compute-bound if SM % >= 80
and >= memory, memory-bound if memory >= 80;
  else latency-bound, named by the largest stall; launch-bound if the kernel is shorter than the median gap between
  launches of its part in the nsys trace.
DRAM bytes are from ncu replays, which flush the caches before each kernel (--cache-control all, ncu's default): they are
cold-L2 bytes, an upper bound on what the kernel moves inside the back-to-back sequence.
Per part (ncu/region_utilisation.csv and c1_bounds.csv rows): every metric weighted by nsys kernel time; the share of
the part's kernel time in each verdict class; coverage = joined nsys time ÷ the part's nsys kernel time; idle fraction =
1 − kernel time ÷ the CUDA-event region time of the T005 row.

python bench/t007_c1_tables.py --model pi0 --mode best_bf16 --c1 results/<d>_E4g_c1_pi0_best_bf16 --nsys results/<d>_E5_pi0_best_bf16_nsys --t005-row <row.json>
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t006_roofline import dim3, join, norm, num  # noqa: E402
from t007_c1 import STALLS  # noqa: E402

PEAK_BW_GBPS = float(json.load(open(os.path.join(os.path.dirname(HERE), "results", "20261022_E1_peak", "hw_constants.json")))["measured_bw_GBps"]["best"]["value"])


def read_raw(path):
    txt = subprocess.run(["zstd", "-dc", path], capture_output=True, text=True).stdout if path.endswith(".zst") else open(path).read()
    rows = list(csv.reader(io.StringIO(txt)))
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[2:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--nsys", required=True)
    ap.add_argument("--t005-row", default="")
    ap.add_argument("--summary-csv", default="", help="append the c1_bounds rows here as well")
    a = ap.parse_args()
    nk = [r for r in csv.DictReader(open(os.path.join(a.nsys, "kernels.csv"))) if r["inference"] == "inference_0"]
    region_ms = {}
    if a.t005_row:
        row = json.load(open(a.t005_row))
        st = row["by_N"][str(row["N"])]["regions"]
        region_ms = {k: st[k]["median_ms"] for k in ("vision", "vlm_prefix", "ae_total") if k in st}
        if "ae_step_individual[0]" in st:
            region_ms["ae_step[0]"] = st["ae_step_individual[0]"]["median_ms"]
    parts = sorted({re.sub(r"_s\d+_raw\.csv(\.zst)?$", "", os.path.basename(p)) for p in glob.glob(os.path.join(a.c1, "ncu", "*_raw.csv*"))})
    krows, bounds = [], []
    for tag in parts:
        part = {"ae_step0": "ae_step[0]"}.get(tag, tag)
        ncu = []
        for p in sorted(glob.glob(os.path.join(a.c1, "ncu", f"{tag}_s*_raw.csv*")), key=lambda p: int(re.search(r"_s(\d+)_raw", p).group(1))):
            ncu += read_raw(p)
        ns = [r for r in nk if r["region"] == part]
        starts = sorted(int(r["start_ns"]) for r in ns)
        ends = sorted(int(r["start_ns"]) + int(r["duration_ns"]) for r in ns)
        gaps = [s2 - e1 for e1, s2 in zip(ends[:-1], starts[1:]) if s2 > e1]
        gap_med = statistics.median(gaps) if gaps else 0
        tot = sum(int(r["duration_ns"]) for r in ns)
        joined, mism, name_only = 0, 0, 0
        for i, j, kind in join(ncu, ns):
            d = ncu[i]
            nsr = ns[j] if j is not None else None
            name = d.get("Kernel Name", "")
            ok = kind == "strict"
            mism += 1 if kind == "none" else 0
            name_only += 1 if kind == "name_only" else 0
            dur = int(nsr["duration_ns"]) if ok else 0
            joined += dur
            sm = num(d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"))
            mem = num(d.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"))
            stalls = {s.split("stalled_")[1].replace("_per_issue_active", ""): num(d.get(s + ".ratio", d.get(s))) for s in STALLS}
            top = max(stalls, key=stalls.get) if stalls else ""
            byt = (num(d.get("lts__d_sectors_fill_sysmem.sum")) + num(d.get("lts__t_sectors_aperture_sysmem_op_write.sum"))) * 32
            dram_pct = (byt / dur) / PEAK_BW_GBPS * 100 if dur else 0.0
            memx = max(mem, dram_pct)           # this chip has no dram__ counters: ncu's memory % omits DRAM (T001)
            if dur and dur < gap_med:
                verdict = "launch-bound"
            elif sm >= 80 and sm >= memx:
                verdict = "compute-bound"
            elif memx >= 80:
                verdict = "memory-bound"
            else:
                verdict = f"latency-bound ({top})"
            krows.append({"model": a.model, "mode": a.mode, "part": part, "idx": i, "launch_idx": nsr["launch_order"] if nsr else "",
                          "join": kind, "kernel": name[:200], "joined": ok,
                          "nsys_ns": dur, "sm_pct": sm, "onchip_mem_pct": mem, "dram_pct_peak_bw": round(dram_pct, 2),
                          "occupancy_pct": num(d.get("sm__warps_active.avg.pct_of_peak_sustained_active")),
                          "tensor_pipe_pct": num(d.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")),
                          "dram_GBps": round(byt / dur, 3) if dur else "", "top_stall": top,
                          "top_stall_share": round(stalls[top] / sum(stalls.values()), 4) if stalls and sum(stalls.values()) else "",
                          "verdict": verdict, **{f"stall_{k}": v for k, v in stalls.items()}})
        ks = [k for k in krows if k["part"] == part and k["joined"] and k["nsys_ns"]]
        w = sum(k["nsys_ns"] for k in ks) or 1
        wavg = lambda key: sum(k[key] * k["nsys_ns"] for k in ks) / w  # noqa: E731
        st_tot = {s: sum(k[f"stall_{s}"] * k["nsys_ns"] for k in ks) / w for s in (x.split("stalled_")[1].replace("_per_issue_active", "") for x in STALLS)}
        top = max(st_tot, key=st_tot.get) if st_tot else ""
        cls = {}
        for k in ks:
            c = k["verdict"].split(" (")[0]
            cls[c] = cls.get(c, 0) + k["nsys_ns"]
        dram = sum((k["dram_GBps"] or 0) * k["nsys_ns"] for k in ks) / w if ks else 0
        ev = region_ms.get(part)
        bounds.append({"model": a.model, "mode": a.mode, "part": part, "n_kernels": len(ns), "n_ncu": len(ncu), "name_mismatches": mism,
                       "join_name_only": name_only,
                       "coverage": round(joined / tot, 4) if tot else 0, "kernel_ms": round(tot / 1e6, 4),
                       "idle_frac": round(1 - (tot / 1e6) / ev, 4) if ev else "", "sm_pct": round(wavg("sm_pct"), 2),
                       "onchip_mem_pct": round(wavg("onchip_mem_pct"), 2), "occupancy_pct": round(wavg("occupancy_pct"), 2),
                       "tensor_pipe_pct": round(wavg("tensor_pipe_pct"), 2), "dram_GBps": round(dram, 2),
                       "pct_peak_bw": round(dram / PEAK_BW_GBPS * 100, 2), "top_stall": top,
                       "top_stall_share": round(st_tot[top] / sum(st_tot.values()), 4) if sum(st_tot.values()) else "",
                       "verdict": max(cls, key=cls.get) if cls else "",
                       **{f"time_share_{c}": round(v / w, 4) for c, v in cls.items()}})
    os.makedirs(os.path.join(a.c1, "ncu"), exist_ok=True)
    for name, rows in (("ncu/kernels.csv", krows), ("ncu/region_utilisation.csv", bounds)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(a.c1, name), "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=keys)
            w_.writeheader()
            w_.writerows(rows)
    print(json.dumps([{k: b[k] for k in ("part", "coverage", "name_mismatches", "verdict", "top_stall", "sm_pct", "onchip_mem_pct", "pct_peak_bw")} for b in bounds], indent=1))


if __name__ == "__main__":
    main()
