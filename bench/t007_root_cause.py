"""T007 step 5 (protocol §10): the gap table and the root-cause chain of every part of one or more model x mode, from the
committed T006 / T007 results of that configuration (latest non-superseded directories):
  E6   results/*_E6_<m>_<mode>*/{roofline.csv, floor.csv, kernel_roofline.csv}  (CUDA-event time, analytic floor, per-kernel
       ncu bytes / FLOPs joined to nsys durations)
  C1   results/*_E4g_c1_<m>_<mode>/ncu/kernels.csv                            (per-kernel stalls, occupancy, verdict)
  E5   results/*_E5_<m>_<mode>_timeline/timeline_summary.csv                  (idle, CPU-starved gaps, graph nodes)
  dims results/<E6 date>_E6_dims/raw/<m>_<mode>/dims.csv                      (GEMM M, K, N)

Gap of a part = CUDA-event time - analytic floor (floor.csv), split exactly into:
  host / idle      event time - Σ kernel durations of the part (nsys, one inference)
  kernels without strict ncu counters   their nsys time (no floor: their counters came from another launch configuration)
  <class> excess   Σ over the class's kernels of (duration - own floor), own floor = max(bytes / BW, FLOPs / dtype peak)
                   at the measured ceilings; classes as bench/t007_timeline.kclass
  serialisation    Σ per-kernel floors - max(Σ bytes / BW, Σ FLOPs / peak): the part-level floor assumes compute and memory
                   work overlap; one stream of dependent kernels cannot (≥ 0)
  measured > analytic   max(Σ bytes / BW, Σ FLOPs / peak) - analytic floor: traffic or work the analytic model does not count
                   (re-reads, cold-L2 bytes, tile padding; negative when the kernels move less, e.g. L2 hits)
The terms add up to the gap by construction. The chain follows the largest term.

python bench/t007_root_cause.py --modes best_bf16 --out results/<date>_E4g_root_cause
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from t007_timeline import kclass  # noqa: E402

MODELS = ["pi0", "pi05", "smolvla", "groot_n17", "groot_n16", "groot_n15"]
NAMES = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA", "groot_n17": "GR00T N1.7", "groot_n16": "GR00T N1.6",
         "groot_n15": "GR00T N1.5"}
SMS = 20


def latest(pattern):
    ds = [d for d in sorted(glob.glob(os.path.join(REPO, "results", pattern)))
          if os.path.exists(os.path.join(d, "meta.json")) and not os.path.exists(os.path.join(d, "SUPERSEDED.md"))]
    return ds[-1] if ds else None


def rel(p):
    return os.path.relpath(p, REPO)


def one(m, mode):
    e6 = latest(f"*_E6_{m}_{mode}*")
    c1 = latest(f"*_E4g_c1_{m}_{mode}")
    tl = latest(f"*_E5_{m}_{mode}_timeline")
    e5 = latest(f"*_E5_{m}_{mode}_nsys*")
    nsk = [r for r in csv.DictReader(open(os.path.join(e5, "kernels.csv"))) if r["inference"] == "inference_0"] if e5 else []
    if not e6:
        return []
    dims_p = os.path.join(REPO, "results", f"{os.path.basename(e6)[:8]}_E6_dims", "raw", f"{m}_{mode}", "dims.csv")
    dims = list(csv.DictReader(open(dims_p))) if os.path.exists(dims_p) else []
    hw = json.load(open(os.path.join(REPO, "results", "20261022_E1_peak", "hw_constants.json")))
    bw = float(hw["measured_bw_GBps"]["best"]["value"])
    event = {r["part"]: float(r["ms"]) for r in csv.DictReader(open(os.path.join(e6, "roofline.csv")))
             if r["source"] == "analytic" and r["time_source"] == "region_cuda_event" and r["ms"]}
    floor = {r["region"]: dict(r, bytes=r["bytes_read"]) for r in csv.DictReader(open(os.path.join(e6, "floor.csv")))
             if r["peak_source"] == "measured"}
    kr = [k for k in csv.DictReader(open(os.path.join(e6, "kernel_roofline.csv"))) if k["ms_nsys"] and k["join"] == "strict"]
    c1k = {}
    if c1:
        for k in csv.DictReader(open(os.path.join(c1, "ncu", "kernels.csv"))):
            c1k[(k["part"], int(k["idx"]))] = k                # C1's ncu order = E6's ncu order (same NVTX range, launch order)
    tls = list(csv.DictReader(open(os.path.join(tl, "timeline_summary.csv")))) if tl else []
    out = []
    for part, ev in event.items():
        if part not in floor or floor[part].get("gap_ratio") in (None, "n/a"):
            continue
        fl = float(floor[part]["floor_ms"])
        ks = [k for k in kr if k["part"] == part]
        if not ks:
            continue
        ksum = sum(float(k["ms_nsys"]) for k in ks)                       # strictly joined kernels
        kall = sum(int(r["duration_ns"]) for r in nsk if r["region"] == part) / 1e6 or ksum
        terms = collections.defaultdict(float)
        cls_k = collections.defaultdict(list)
        own_sum, sum_b, sum_fc = 0.0, 0.0, 0.0
        for k in ks:
            t = float(k["ms_nsys"])
            byt = float(k["bytes_ncu"] or 0)
            fk = max(byt / (bw * 1e9), float(k["flops_ncu"] or 0) / (float(k["ceiling_TFLOPps"] or 1) * 1e12)) * 1e3
            own_sum += fk
            sum_b += byt
            sum_fc += float(k["flops_ncu"] or 0) / (float(k["ceiling_TFLOPps"] or 1) * 1e12) * 1e3
            c = kclass(k["kernel"])
            terms[f"{c} excess"] += t - fk
            cls_k[c].append((k, t, fk, byt))
        terms["host / idle"] = ev - kall
        terms["kernels without strict ncu counters"] = kall - ksum
        meas_floor = max(sum_b / (bw * 1e9) * 1e3, sum_fc)
        terms["serialisation"] = own_sum - meas_floor          # >= 0: Σ max ≥ max Σ
        terms["measured > analytic"] = meas_floor - fl
        gap = ev - fl
        rows = [{"model": m, "mode": mode, "part": part, "event_ms": round(ev, 4), "floor_ms": round(fl, 4), "gap_ms": round(gap, 4),
                 "factor": f, "contribution_ms": round(v, 4), "share_of_gap": round(v / gap, 4) if gap else ""}
                for f, v in sorted(terms.items(), key=lambda x: -x[1])]
        top = rows[0]["factor"]
        # ---- the chain for the largest term
        src_e6, src_c1, src_tl = rel(e6), rel(c1) if c1 else "—", rel(tl) if tl else "—"
        chain = [("symptom", f"`{part}` takes {ev:.2f} ms (CUDA event) against an analytic floor of {fl:.2f} ms "
                             f"({floor[part]['bound']}-bound floor): {ev / fl:.2f}× the floor, gap {gap:.2f} ms",
                  f"`{src_e6}/floor.csv`, `roofline.csv`")]
        if top == "host / idle":
            reg = [r for r in tls if r["region"] == part or (part.startswith("ae_step") and r["region"].startswith("ae_step["))]
            mean = lambda k: statistics.mean(float(r[k]) for r in reg if r.get(k) not in ("", None)) if reg else float("nan")  # noqa: E731
            chain += [("why 1", f"the GPU executes kernels for only {kall:.2f} of the {ev:.2f} ms; {ev - kall:.2f} ms "
                                f"({(ev - kall) / gap * 100:.0f} % of the gap) no kernel runs",
                       f"`{src_e6}/kernel_roofline.csv` (Σ nsys durations), floor.csv"),
                      ("why 2", f"{mean('starved_frac_of_gaps') * 100:.0f} % of the idle gap time is CPU-starved: the next kernel's "
                                f"launch call returned only after the previous kernel had finished",
                       f"`{src_tl}/timeline_summary.csv` (starved_frac_of_gaps)"),
                      ("why 3", f"the part launches {mean('kernels'):.0f} kernels per {'step' if part.startswith('ae_step') else 'inference'}"
                                f", {mean('graph_nodes'):.0f} of them from CUDA graphs; the rest are issued one by one from Python",
                       f"`{src_tl}/timeline_summary.csv` (kernels, graph_nodes), `cpu_thread.csv`"),
                      ("root cause", "host launch overhead: the part's kernels are issued individually by the CPU faster than "
                                     "the GPU can be kept busy (eager / non-graphed execution)", "")]
        elif top.endswith(" excess"):
            c = top[:-7]
            kk = cls_k[c]
            t_c = sum(t for _, t, _, _ in kk)
            b_c = sum(b for _, _, _, b in kk)
            gbps = b_c / (t_c / 1e3) / 1e9 if t_c else 0
            joined = [(k, t) for k, t, _, _ in kk if (part, int(k["ncu_idx"])) in c1k]
            w = sum(t for _, t in joined) or 1
            occ = sum(float(c1k[(part, int(k["ncu_idx"]))]["occupancy_pct"] or 0) * t for k, t in joined) / w
            st = collections.Counter()
            for k, t in joined:
                r = c1k[(part, int(k["ncu_idx"]))]
                for key, v in r.items():
                    if key.startswith("stall_") and v not in ("", None):
                        st[key[6:]] += float(v) * t
            sttot = sum(st.values()) or 1
            top_st = st.most_common(2)
            grids = collections.Counter()
            for k, t, _, _ in kk:
                g = 1
                for v in k["grid"].split(","):
                    g *= int(v)
                grids[g] += t
            gmain, gshare = grids.most_common(1)[0] if grids else (0, 0)
            ms_ = sorted(int(d["M"]) for d in dims if d["region"] == part or (part == "ae_total" and d["region"].startswith(("ae_step", "ae_dit"))))
            m_med = ms_[len(ms_) // 2] if ms_ else None
            chain += [("why 1", f"{c} kernels run {t_c:.2f} ms but would need {sum(f for _, _, f, _ in kk):.2f} ms at the measured "
                                f"ceilings: {terms[top]:.2f} ms ({terms[top] / gap * 100:.0f} % of the gap); they move "
                                f"{b_c / 1e6:.1f} MB at {gbps:.0f} GB/s = {gbps / bw * 100:.0f} % of {bw:.0f} GB/s",
                       f"`{src_e6}/kernel_roofline.csv` (dram_read_B + dram_write_B, flops, nsys_duration_ns)"),
                      ("why 2", (f"inside them the warps wait: top stalls {top_st[0][0]} {top_st[0][1] / sttot * 100:.0f} %"
                                 + (f", {top_st[1][0]} {top_st[1][1] / sttot * 100:.0f} %" if len(top_st) > 1 else "")
                                 + f" of stall cycles; achieved occupancy {occ:.1f} % (time-weighted)") if joined else
                                "C1 not captured for this configuration",
                       f"`{src_c1}/ncu/kernels.csv` (stall_*, occupancy_pct)"),
                      ("why 3", f"the dominant launch grid is {gmain} thread blocks ({gshare / t_c * 100:.0f} % of the class's time) "
                                f"on {SMS} SMs" + (f"; the GEMMs multiply M = {m_med} rows (median)" if m_med and c == "GEMM" else "")
                                + ": too few warps in flight to cover DRAM latency", f"`{src_e6}/kernel_roofline.csv` (grid), "
                                f"`{rel(dims_p) if dims else '—'}` (M)"),
                      ("root cause", (f"the {c} kernels are too small for the device: M = {m_med} rows and {gmain}-block grids "
                                      "leave most warp slots empty, so memory latency is exposed and DRAM runs below its peak")
                                     if c == "GEMM" else
                                     f"the {c} kernels run far below the device's ceilings (small, latency-exposed launches)", "")]
        elif top == "kernels without strict ncu counters":
            chain += [("why 1", f"{kall - ksum:.2f} ms ({terms[top] / gap * 100:.0f} % of the gap) is spent in kernels whose ncu row "
                                "matched by name but not by grid / block, so their counters cannot be used", f"`{src_e6}/coverage.json`"),
                      ("why 2", "the profiled process re-tuned those Inductor kernels and launched them with another block size",
                       f"`{src_e6}/kernel_roofline.csv` (join = name_only)"),
                      ("why 3", "Inductor's runtime autotuning results were not reused by the profiling process", "Inductor cache"),
                      ("root cause", "a measurement gap, not a model property: re-capture with the timed run's tuned configs", "")]
        elif top == "serialisation":
            mem_t = sum(float(k["ms_nsys"]) for k in ks if k["bound_by_intensity"] == "memory")
            cmp_t = sum(float(k["ms_nsys"]) for k in ks if k["bound_by_intensity"] == "compute")
            chain += [("why 1", f"each kernel's own floor, summed, is {own_sum:.2f} ms, but the part's totals would allow "
                                f"{meas_floor:.2f} ms if compute and memory work overlapped: {terms[top]:.2f} ms "
                                f"({terms[top] / gap * 100:.0f} % of the gap)", f"`{src_e6}/kernel_roofline.csv`"),
                      ("why 2", f"the part alternates compute-side kernels ({cmp_t:.2f} ms of kernel time) and memory-side kernels "
                                f"({mem_t:.2f} ms); while one class runs, the other resource idles",
                       f"`{src_e6}/kernel_roofline.csv` (bound_by_intensity, nsys_duration_ns)"),
                      ("why 3", "the kernels execute one after another on one stream (a layer's GEMMs, then its norms and "
                                "elementwise ops); nothing runs concurrently", f"`{src_tl}/kernels_per_region.csv`, nsys trace order"),
                      ("root cause", "serial execution of kernels with different bottlenecks: the part-level roofline floor "
                                     "assumes an overlap that a single stream of dependent kernels cannot give", "")]
        else:
            fl_r = floor[part]
            b_an, f_an = float(fl_r["bytes"]), float(fl_r["flops"])
            f_meas = sum(float(k["flops_ncu"] or 0) for k in ks)
            regs = {part} if part != "ae_total" else {r["region"] for r in dims if r["region"].startswith(("ae_step", "ae_dit"))}
            nrep = 1 if part != "ae_total" else len([k for k in floor if k.startswith("ae_step[")]) or 4
            act_b = sum((int(r["act_bytes"]) + int(r["out_bytes"])) * int(r["calls_per_inference"]) for r in dims
                        if r["region"] in regs) * (nrep if part == "ae_total" else 1)
            chain += [("why 1", f"the kernels move {sum_b / 1e9:.3f} GB and execute {f_meas / 1e12:.3f} TFLOP against the analytic "
                                f"{b_an / 1e9:.3f} GB and {f_an / 1e12:.3f} TFLOP ({sum_b / b_an if b_an else float('nan'):.2f}× and "
                                f"{f_meas / f_an if f_an else float('nan'):.2f}×): "
                                f"{terms[top]:.2f} ms ({terms[top] / gap * 100:.0f} % of the gap)",
                       f"`{src_e6}/kernel_roofline.csv`, floor.csv"),
                      ("why 2", f"the analytic floor counts parameters and the hand-over only; the activations the part's GEMMs "
                                f"read and write add {act_b / 1e9:.3f} GB (C3 dims) that it leaves out, and every kernel writes its "
                                "output for the next one to read back", f"`{rel(dims_p) if dims else '—'}` (act_bytes, out_bytes), "
                                "bench/t006_analytic.py"),
                      ("why 3", "ncu flushes the caches before each kernel it replays (cold L2), so a tensor the next kernel "
                                "would find in the 32 MiB L2 is counted as DRAM traffic; tensor FLOPs are counted as executed, "
                                "with tile padding", "bench/t006_capture.py (ncu defaults); T003 counter validation"),
                      ("root cause", "activation round trips between unfused kernels, beyond the weight-only minimum; the "
                                     "cold-L2 share of the measured bytes is an upper bound, not run-time traffic", "")]
        out.append({"model": m, "mode": mode, "part": part, "rows": rows, "chain": chain,
                    "sources": {"E6": src_e6, "C1": src_c1, "E5": src_tl, "dims": rel(dims_p) if dims else None}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="best_bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = [r for mode in a.modes.split(",") for m in MODELS for r in one(m, mode)]
    rows = [x for r in res for x in r["rows"]]
    with open(os.path.join(a.out, "gap_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump([{k: r[k] for k in ("model", "mode", "part", "chain", "sources")} for r in res],
              open(os.path.join(a.out, "chains.json"), "w"), indent=1, ensure_ascii=False)
    L = []
    for r in res:
        L += [f"### {NAMES[r['model']]} · {r['mode']} · `{r['part']}`", "", "| step | observation | evidence |", "|---|---|---|"]
        L += [f"| {s} | {o} | {e} |" for s, o, e in r["chain"]]
        L += ["", "| factor | contribution ms | share of gap |", "|---|---|---|"]
        L += [f"| {x['factor']} | {x['contribution_ms']:.2f} | {x['share_of_gap'] * 100:.0f} % |" for x in r["rows"]
              if x["share_of_gap"] != ""]
        L.append("")
    open(os.path.join(a.out, "root_cause.md"), "w").write("\n".join(L) + "\n")
    print(f"{len(res)} parts -> {a.out}")


if __name__ == "__main__":
    main()
