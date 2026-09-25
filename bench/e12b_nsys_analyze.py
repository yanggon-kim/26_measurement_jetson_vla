"""Analyse the E12b nsys CSV exports -> corun_breakdown.csv and the prefix grid-size table.

Inputs: `nsys stats --report cuda_gpu_trace --format csv` and `--report nvtx_pushpop_trace`.
For each NVTX range (pipeline, prefix_alone, expert_alone, serial):
  * per stream busy time (union of kernel intervals) and, in `pipeline`, the time during which
    BOTH streams have a kernel executing -> the fraction of the frame that overlaps;
  * the co-run decomposition (M2 step 5): for each stage, (a) kernel-duration inflation = sum of
    its kernel durations co-run minus isolated, per frame; (b) inter-kernel gaps inside the stage
    while the OTHER stream is executing = queueing behind the other stage's grids (KerColle's
    head-of-line blocking if large); (c) the remainder of the stage's span = host issue delay /
    idle.
  * the prefix's kernels with grid sizes against what 16 SMs hold at once (one CTA per SM is 16;
    the table lists CTAs so the reader can apply occupancy).
Stream identity comes from the isolated ranges: the stream active in prefix_alone is the prefix
stream, in expert_alone the expert's.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(p):
    lines = Path(p).read_text(errors="replace").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"Start') or l.startswith("Start"))
    return list(csv.DictReader(lines[i:]))


def col(r, *names):
    for n in names:
        for k in r:
            if k.strip('"').startswith(n):
                return r[k]
    return None


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def merged(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def inter(a, b):
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-trace", required=True)
    ap.add_argument("--nvtx", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    kern = []
    for r in read_csv(args.gpu_trace):
        name = col(r, "Name") or ""
        if "memcpy" in name.lower() or "memset" in name.lower():
            kind = "mem"
        else:
            kind = "kernel"
        try:
            st = int(float(col(r, "Start")))
            du = int(float(col(r, "Duration")))
        except (TypeError, ValueError):
            continue
        grid = [col(r, "GrdX"), col(r, "GrdY"), col(r, "GrdZ")]
        try:
            ctas = int(grid[0]) * int(grid[1]) * int(grid[2])
        except (TypeError, ValueError):
            ctas = None
        kern.append({"start": st, "end": st + du, "dur": du, "stream": col(r, "Strm"),
                     "name": name, "kind": kind, "ctas": ctas})
    ranges = {}
    for r in read_csv(args.nvtx):
        # nsys prefixes pushed-range names with ':' (":pipeline"); only the top-level ranges
        # this harness pushed are wanted, not the CCCL/library ranges nested inside them.
        nm = (col(r, "Name") or "").strip('"').lstrip(":")
        if str(col(r, "Lvl")).strip() not in ("0", ""):
            continue
        try:
            ranges[nm] = (int(float(col(r, "Start"))), int(float(col(r, "End"))))
        except (TypeError, ValueError):
            continue

    def in_range(nm):
        s, e = ranges[nm]
        return [k for k in kern if k["start"] >= s and k["end"] <= e + 5_000_000]

    def main_stream(nm):
        c = defaultdict(int)
        for k in in_range(nm):
            c[k["stream"]] += k["dur"]
        return max(c, key=c.get) if c else None

    sp, se = main_stream("prefix_alone"), main_stream("expert_alone")
    rows = []
    for nm, F in (("pipeline", 5), ("prefix_alone", 5), ("expert_alone", 5), ("serial", 3)):
        if nm not in ranges:
            continue
        ks = in_range(nm)
        span = ranges[nm][1] - ranges[nm][0]
        P = merged([(k["start"], k["end"]) for k in ks if k["stream"] == sp])
        E = merged([(k["start"], k["end"]) for k in ks if k["stream"] == se])
        both = inter(P, E)
        rows.append({"range": nm, "frames": F, "span_ms": span / 1e6,
                     "prefix_stream_busy_ms": sum(e - s for s, e in P) / 1e6,
                     "expert_stream_busy_ms": sum(e - s for s, e in E) / 1e6,
                     "both_busy_ms": both / 1e6, "both_busy_frac_of_span": both / span if span else 0,
                     "prefix_kernel_sum_ms": sum(k["dur"] for k in ks if k["stream"] == sp) / 1e6,
                     "expert_kernel_sum_ms": sum(k["dur"] for k in ks if k["stream"] == se) / 1e6,
                     "prefix_kernels": sum(1 for k in ks if k["stream"] == sp),
                     "expert_kernels": sum(1 for k in ks if k["stream"] == se)})
    by = {r["range"]: r for r in rows}
    dec = []
    if "pipeline" in by and "prefix_alone" in by and "expert_alone" in by:
        pl, pa, ea = by["pipeline"], by["prefix_alone"], by["expert_alone"]
        for stage, key, iso in (("prefix", "prefix", pa), ("expert", "expert", ea)):
            per_co = pl[f"{key}_kernel_sum_ms"] / pl["frames"]
            per_iso = iso[f"{key}_kernel_sum_ms"] / iso["frames"]
            busy_co = pl[f"{key}_stream_busy_ms"] / pl["frames"]
            busy_iso = iso[f"{key}_stream_busy_ms"] / iso["frames"]
            dec.append({"stage": stage, "kernel_sum_iso_ms_per_frame": round(per_iso, 4),
                        "kernel_sum_corun_ms_per_frame": round(per_co, 4),
                        "kernel_duration_inflation_ms": round(per_co - per_iso, 4),
                        "stream_busy_iso_ms_per_frame": round(busy_iso, 4),
                        "stream_busy_corun_ms_per_frame": round(busy_co, 4),
                        "note": "inflation = the same kernels running longer (shared SMs, L2, "
                                "DRAM); busy-vs-sum difference = kernels of this stage "
                                "overlapping each other"})
    # queueing: gaps inside each stage's kernels while the other stream executes, pipeline range
    if "pipeline" in ranges:
        ks = in_range("pipeline")
        for stage, s_self, s_other in (("prefix", sp, se), ("expert", se, sp)):
            mine = sorted((k for k in ks if k["stream"] == s_self), key=lambda k: k["start"])
            other = merged([(k["start"], k["end"]) for k in ks if k["stream"] == s_other])
            gaps = [[a["end"], b["start"]] for a, b in zip(mine, mine[1:]) if b["start"] > a["end"]]
            g_tot = sum(e - s for s, e in gaps)
            g_other = inter(merged(gaps), other)
            for d in dec:
                if d["stage"] == stage:
                    d["gaps_ms_per_frame"] = round(g_tot / 1e6 / 5, 4)
                    d["gaps_while_other_stream_busy_ms_per_frame"] = round(g_other / 1e6 / 5, 4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "corun_breakdown.csv").open("w", newline="") as f:
        keys = list(rows[0]) if rows else []
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    if dec:
        keys = []
        for d in dec:
            for k in d:
                if k not in keys:
                    keys.append(k)
        with (out / "corun_decomposition.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(dec)
    # prefix grid sizes vs 16 SMs
    pk = [k for k in in_range("prefix_alone") if k["stream"] == sp and k["kind"] == "kernel"] \
        if "prefix_alone" in ranges else []
    agg = defaultdict(lambda: {"launches": 0, "dur": 0, "ctas": set()})
    for k in pk:
        a = agg[k["name"][:90]]
        a["launches"] += 1; a["dur"] += k["dur"]
        if k["ctas"] is not None:
            a["ctas"].add(k["ctas"])
    tot = sum(a["dur"] for a in agg.values()) or 1
    with (out / "prefix_grids.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "launches", "time_ms_per_frame",
                                          "share_of_prefix_kernel_time", "ctas",
                                          "waves_at_1_cta_per_sm_on_16"])
        w.writeheader()
        for n, a in sorted(agg.items(), key=lambda kv: -kv[1]["dur"])[:40]:
            c = max(a["ctas"]) if a["ctas"] else None
            w.writerow({"kernel": n, "launches": a["launches"] // 5,
                        "time_ms_per_frame": round(a["dur"] / 1e6 / 5, 4),
                        "share_of_prefix_kernel_time": round(a["dur"] / tot, 4),
                        "ctas": c, "waves_at_1_cta_per_sm_on_16": round(c / 16, 2) if c else ""})
    for r in rows:
        print(f"  {r['range']:13s} span {r['span_ms']:8.2f} ms  prefix busy {r['prefix_stream_busy_ms']:8.2f}"
              f"  expert busy {r['expert_stream_busy_ms']:8.2f}  both {r['both_busy_ms']:8.2f} "
              f"({100 * r['both_busy_frac_of_span']:.1f} % of span)")
    for d in dec:
        print("  ", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
