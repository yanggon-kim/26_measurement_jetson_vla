"""T007 step 1: E5 timeline analysis of one model x mode from T006's nsys capture (results/<D>_E5_<m>_<mode>_nsys*).

GPU side, from the committed exports (kernels.csv; nsys/*_nvtx_gpu_proj_trace.csv):
  per inference x region instance (vision, vlm_prefix, backbone_lm_head, vlm_backbone, model_get_action, ae_total,
  ae_step[i], ae_dit[i], inference_k): wall = the GPU-projected NVTX range, kernels whose start lies in it, summed kernel
  time, kernel-interval UNION, GPU idle = wall - union (reference/orin/bench/parse_nsys.py's definition), and the same
  idle against the untraced CUDA-event time of the T005 row (nsys tracing stretches ranges, so both are given with
  their denominators), gaps between consecutive kernels, kernel classes by time, graph nodes vs direct launches.
CPU side, from the capture's SQLite export (scratch/, not committed: nsys files embed the environment, T001):
  the launching thread of each region (most CUDA runtime calls inside the CPU range), its CUDA API time, launch /
  graph-launch / synchronize calls, its on-CPU time from the scheduler trace (SCHED_EVENTS), and the starved-gap
  gap split: a gap between two nodes of one graph launch (same correlation id) is intra-graph (GPU-side); otherwise
  it is CPU-starved when kernel k's launch call returned after kernel k-1 had finished (the GPU had nothing queued),
  else queued (the work was submitted; the gap is launch latency or a dependency on the device).
  starved_frac = starved gap time / all gap time.

Kernel classes (first rule that matches the base name): attention (flash|fmha|attention|sdpa|softmax), GEMM (nvjet|
gemm|cutlass|xmma|sgemm|convolve|triton_tem_*|triton_red/per_* with mm/addmm/bmm), norm (layer_norm|rms|rsqrt|
group_norm), copy (copy kernels, cat, and triton_poi kernels made only of view/layout ops), elementwise (elementwise|
triton_poi_*|multi_tensor_apply), other.

python bench/t007_timeline.py --e5 results/<D>_E5_<m>_<mode>_nsys --t005-row <row.json> --out results/<D>_E5_<m>_<mode>_timeline
"""
import argparse
import bisect
import collections
import csv
import glob
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from t006_roofline import norm  # noqa: E402

REGIONS = ("vision", "vlm_prefix", "backbone_lm_head", "vlm_backbone", "model_get_action", "ae_total")
LAYOUT_OPS = {"clone", "transpose", "view", "cat", "permute", "unsafe", "slice", "copy", "to", "expand", "unsqueeze",
              "index", "select", "squeeze", "split", "constant", "pad", "nd", "t", "stack", "reshape", "contiguous"}
GAP_BUCKETS_US = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1e9]


def kclass(name):
    n = norm(name)
    low = n.lower()
    if re.search(r"flash|fmha|attention|sdpa|softmax", low):
        return "attention"
    if re.search(r"nvjet|gemm|cutlass|xmma|sgemm|convolve", low) or low.startswith("triton_tem_") or \
            (re.match(r"triton_(red|per)_", low) and re.search(r"_(addmm|mm|bmm)(_|$)", low)):
        return "GEMM"
    if re.search(r"layer_norm|rms|rsqrt|group_norm", low):
        return "norm"
    if re.search(r"catarraybatchedcopy|copy_kernel|memcpy|^cat", low):
        return "copy"
    if low.startswith("triton_poi_"):
        ops = set(re.sub(r"^triton_poi_fused_?", "", low).split("_")) - {""}
        ops = {o for o in ops if not o.isdigit()}
        return "copy" if ops and ops <= LAYOUT_OPS else "elementwise"
    if re.search(r"elementwise|multi_tensor_apply", low):
        return "elementwise"
    return "other"


def union(ivs):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(ivs):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


def sqlite_of(e5):
    base = os.path.basename(e5.rstrip("/"))
    hits = glob.glob(os.path.join(REPO, "scratch", base, "*.sqlite"))
    if hits:
        return hits[0]
    reps = glob.glob(os.path.join(REPO, "scratch", base, "*.nsys-rep"))
    if not reps:
        return None
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", reps[0][:-9] + ".sqlite", reps[0]],
                   check=True, stdout=subprocess.DEVNULL)
    return reps[0][:-9] + ".sqlite"


class Cpu:
    """CUDA runtime calls, kernel launch correlation and scheduler intervals from the nsys SQLite."""

    def __init__(self, path):
        c = sqlite3.connect(path)
        names = dict(c.execute("select id, value from StringIds"))
        self.api = collections.defaultdict(list)          # tid -> [(start, end, name)]
        self.api_end = {}                                 # correlationId -> end of the launching call
        for s, e, tid, corr, nid in c.execute("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
            self.api[tid].append((s, e, names.get(nid, "")))
            self.api_end[corr] = e
        for v in self.api.values():
            v.sort()
        self.kern = {}                                    # (start, end) -> (correlationId, graphNodeId)
        for s, e, corr, g in c.execute("select start, end, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"):
            self.kern[(s, e)] = (corr, g)
        self.sched = collections.defaultdict(list)        # tid -> [(t, is_in)]
        for t, tid, isin in c.execute("select start, globalTid, isSchedIn from SCHED_EVENTS order by start"):
            self.sched[tid].append((t, bool(isin)))
        self.tid_of = {}
        for (gt,) in c.execute("select distinct globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
            self.tid_of[gt & 0xFFFFFF] = gt                # nsys globalTid low 24 bits = OS tid

    def on_cpu_ns(self, gtid, lo, hi):
        ev = self.sched.get(gtid, [])
        ts = [t for t, _ in ev]
        i = bisect.bisect_left(ts, lo)
        state = ev[i - 1][1] if i else True               # no switch before the range: running (it issued the range)
        t, tot = lo, 0
        for tt, isin in ev[i:]:
            if tt >= hi:
                break
            if state:
                tot += tt - t
            state, t = isin, tt
        return tot + ((hi - t) if state else 0)

    def calls(self, gtid, lo, hi):
        v = self.api.get(gtid, [])
        i = bisect.bisect_left(v, (lo, 0, ""))
        out = []
        for s, e, n in v[i:]:
            if s >= hi:
                break
            out.append((s, e, n))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", required=True)
    ap.add_argument("--t005-row", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.e5, "raw", "target_info.json")))
    kern = [r for r in csv.DictReader(open(os.path.join(a.e5, "kernels.csv")))]
    for r in kern:
        r["s"], r["e"] = int(r["start_ns"]), int(r["start_ns"]) + int(r["duration_ns"])
        r["cls"] = kclass(r["name"])
    kern.sort(key=lambda r: r["s"])
    starts = [r["s"] for r in kern]
    proj = list(csv.DictReader(open(glob.glob(os.path.join(a.e5, "nsys", "*_nvtx_gpu_proj_trace.csv"))[0])))
    ev_ms = {}
    if a.t005_row:
        row = json.load(open(a.t005_row))
        st = row["by_N"][str(row["N"])]["regions"]
        ev_ms = {k: v["median_ms"] for k, v in st.items() if isinstance(v, dict) and "median_ms" in v}
        if "e2e_wall" in ev_ms:
            ev_ms["inference"] = ev_ms["e2e_wall"]
    sq = sqlite_of(a.e5)
    cpu = Cpu(sq) if sq else None
    inst = collections.defaultdict(list)                  # (inference, region) -> [(gpu_s, gpu_e, cpu_s, cpu_e, tid)]
    cur_inf = None
    for r in sorted(proj, key=lambda r: int(r["Orig Start (ns)"] or 0)):
        name = r["Name"].lstrip(":")
        if not r["Projected Start (ns)"]:
            continue
        if name.startswith("inference_"):
            cur_inf = name
        if not (name in REGIONS or name.startswith(("ae_step[", "ae_dit[", "inference_"))):
            continue
        gs = int(r["Projected Start (ns)"])
        cs = int(r["Orig Start (ns)"])
        key = "inference" if name.startswith("inference_") else name
        inst[(cur_inf, key)].append((gs, gs + int(r["Projected Duration (ns)"]), cs, cs + int(r["Orig Duration (ns)"]), int(r["TID"])))
    srows, grows, krows, crows, clsrows = [], [], [], [], []
    for (inf, reg), ivs in sorted(inst.items()):
        ks, gaps, starved, nstarved, intra = [], [], 0, 0, 0
        for gs, ge, _, _, _ in ivs:
            i = bisect.bisect_left(starts, gs)
            inside = []
            while i < len(kern) and kern[i]["s"] < ge:
                inside.append(kern[i])
                i += 1
            ks += inside
            for k0, k1 in zip(inside[:-1], inside[1:]):
                g = k1["s"] - max(k0["e"], k0["s"])
                if g <= 0:
                    continue
                gaps.append(g)
                if cpu:
                    c0 = cpu.kern.get((k0["s"], k0["e"]), (None, None))[0]
                    c1 = cpu.kern.get((k1["s"], k1["e"]), (None, None))[0]
                    api_e = cpu.api_end.get(c1)
                    if c1 is not None and c1 == c0:
                        intra += g                        # two nodes of one graph launch: GPU-side node-to-node gap
                    elif api_e is not None and api_e > k0["e"]:
                        starved += g                      # the launch call returned after the GPU had run dry
                        nstarved += 1
        wall = sum(ge - gs for gs, ge, *_ in ivs)
        un = sum(union([(k["s"], min(k["e"], ge)) for k in ks if gs <= k["s"] < ge]) for gs, ge, *_ in ivs)
        ksum = sum(int(k["duration_ns"]) for k in ks)
        evk = reg if reg in ev_ms else None
        if reg.startswith("ae_step[") and f"ae_step_individual[{reg[8:-1]}]" in ev_ms:
            evk = f"ae_step_individual[{reg[8:-1]}]"
        ev = ev_ms.get(evk) if evk else None
        graph_nodes = sum(1 for k in ks if cpu and cpu.kern.get((k["s"], k["e"]), (None, None))[1])
        srows.append({"inference": inf, "region": reg, "instances": len(ivs), "wall_ms_nsys": round(wall / 1e6, 4),
                      "kernels": len(ks), "graph_nodes": graph_nodes if cpu else "",
                      "direct_launches": (len(ks) - graph_nodes) if cpu else "",
                      "kernel_sum_ms": round(ksum / 1e6, 4), "kernel_union_ms": round(un / 1e6, 4),
                      "gpu_idle_ms_nsys": round((wall - un) / 1e6, 4), "gpu_idle_frac_nsys": round((wall - un) / wall, 4) if wall else "",
                      "event_ms_t005": ev if ev is not None else "",
                      "gpu_idle_frac_vs_event": round(1 - (un / 1e6) / ev, 4) if ev else "",
                      "nsys_stretch": round((wall / 1e6) / ev, 4) if ev else "",
                      "n_gaps": len(gaps), "gap_total_ms": round(sum(gaps) / 1e6, 4),
                      "gap_median_us": round(statistics.median(gaps) / 1e3, 3) if gaps else "",
                      "gap_p95_us": round(sorted(gaps)[int(0.95 * (len(gaps) - 1))] / 1e3, 3) if gaps else "",
                      "starved_gaps": nstarved if cpu else "", "starved_gap_ms": round(starved / 1e6, 4) if cpu else "",
                      "starved_frac_of_gaps": round(starved / sum(gaps), 4) if cpu and gaps else "",
                      "intra_graph_gap_ms": round(intra / 1e6, 4) if cpu else "",
                      "queued_gap_ms": round((sum(gaps) - starved - intra) / 1e6, 4) if cpu else ""})
        for lo, hi in zip(GAP_BUCKETS_US[:-1], GAP_BUCKETS_US[1:]):
            sel = [g for g in gaps if lo * 1e3 <= g < hi * 1e3]
            grows.append({"inference": inf, "region": reg, "gap_bucket_us": f"{lo}-{'inf' if hi > 1e8 else hi}",
                          "count": len(sel), "total_us": round(sum(sel) / 1e3, 3)})
        per = collections.defaultdict(lambda: [0, 0])
        for k in ks:
            p = per[(norm(k["name"])[:160], k["cls"])]
            p[0] += 1
            p[1] += int(k["duration_ns"])
        for (n, c), (cnt, d) in sorted(per.items(), key=lambda x: -x[1][1]):
            krows.append({"inference": inf, "region": reg, "kernel": n, "class": c, "launches": cnt,
                          "total_ms": round(d / 1e6, 4), "share": round(d / ksum, 4) if ksum else ""})
        cl = collections.Counter()
        for k in ks:
            cl[k["cls"]] += int(k["duration_ns"])
        for c in ("GEMM", "attention", "norm", "elementwise", "copy", "other"):
            clsrows.append({"inference": inf, "region": reg, "class": c, "ms": round(cl[c] / 1e6, 4),
                            "share": round(cl[c] / ksum, 4) if ksum else ""})
        if cpu:
            for gs, ge, cs, ce, tid in ivs:
                gt = cpu.tid_of.get(tid)
                calls = cpu.calls(gt, cs, ce) if gt else []
                n = lambda pat: sum(1 for c in calls if re.search(pat, c[2]))  # noqa: E731
                crows.append({"inference": inf, "region": reg, "tid": tid, "cpu_range_ms": round((ce - cs) / 1e6, 4),
                              "gpu_range_ms": round((ge - gs) / 1e6, 4),
                              "cuda_api_calls": len(calls), "cuda_api_ms": round(sum(e - s for s, e, _ in calls) / 1e6, 4),
                              "kernel_launch_calls": n(r"LaunchKernel|cuLaunch"), "graph_launch_calls": n(r"GraphLaunch"),
                              "sync_calls": n(r"Synchronize|EventSynchronize|StreamWait"),
                              "sync_ms": round(sum(e - s for s, e, nm in calls if re.search(r"Synchronize", nm)) / 1e6, 4),
                              "on_cpu_ms": round(cpu.on_cpu_ns(gt, cs, ce) / 1e6, 4) if gt else "",
                              "on_cpu_frac": round(cpu.on_cpu_ns(gt, cs, ce) / (ce - cs), 4) if gt and ce > cs else "",
                              "cpu_end_minus_gpu_end_ms": round((ce - ge) / 1e6, 4)})
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("timeline_summary.csv", srows), ("gaps.csv", grows), ("kernels_per_region.csv", krows),
                       ("kernel_classes.csv", clsrows), ("cpu_thread.csv", crows)):
        if not rows:
            continue
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump({"e5_capture": os.path.relpath(a.e5, REPO), "sqlite": os.path.relpath(sq, REPO) if sq else None,
               "t005_row": os.path.relpath(a.t005_row, REPO) if a.t005_row else None, "rung": info.get("rung"),
               "class_rules": __doc__.split("Kernel classes")[1].split("python bench")[0].strip()},
              open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    for r in srows:
        if r["inference"] == "inference_1" and not r["region"].startswith(("ae_step[", "ae_dit[")) or r["region"] == "ae_step[0]":
            print({k: r[k] for k in ("inference", "region", "wall_ms_nsys", "kernels", "graph_nodes", "kernel_union_ms",
                                     "gpu_idle_frac_nsys", "event_ms_t005", "gpu_idle_frac_vs_event", "starved_frac_of_gaps", "intra_graph_gap_ms", "queued_gap_ms")})


if __name__ == "__main__":
    main()
