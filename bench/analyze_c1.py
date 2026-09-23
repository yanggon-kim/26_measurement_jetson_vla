"""C1 + C2 — what each phase is bound by, and what the best rung actually fixed.

C1 asks, per model and per phase: memory bound, compute bound, or latency bound, and on
which stall. C2 asks what changed between the eager and the best configuration, phase by
phase, so an optimisation can be matched to the bottleneck it removed.

Both are answered from the ncu CSVs that bench/run_ncu_c1*.sh export. One capture is one
(model, mode, region): 40 kernel launches, each with Duration, Compute (SM) Throughput,
Memory Throughput, Achieved Occupancy and the 14
smsp__average_warps_issue_stalled_*_per_issue_active.ratio metrics.

## How a phase is summarised

Every per-kernel number is averaged **weighted by that kernel's duration**. An unweighted
mean would let a 2 us elementwise kernel count as much as a 234 us convolution, which is
how a phase that is plainly memory bound can be made to look balanced.

## How "bound by" is decided, and where the thresholds come from

ncu's own SOLBottleneck rule fires at 80 % of either SM or memory throughput, and that is
the threshold used here rather than one invented for this study:

    max(SM%, Mem%) >= 80            -> bound by whichever is higher
    both < 60                       -> latency bound; the dominant stall names which latency
    otherwise                       -> mixed, and both numbers are reported

"Long Scoreboard" is called out separately wherever it leads, because it is the signature
of waiting on global memory and it is the specific thing this study set out to test: the
research premise is that the action expert is memory bound.

    python bench/analyze_c1.py --out results/20260923_E4g_c1_summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"

SCALARS = {
    "Duration": "duration",
    "Compute (SM) Throughput": "sm_pct",
    "Memory Throughput": "mem_pct",
    "Achieved Occupancy": "occupancy_pct",
    "L2 Hit Rate": "l2_hit_pct",
    "L1/TEX Hit Rate": "l1_hit_pct",
}

# ncu reports Duration in us or ms depending on the kernel; normalise to us.
_DUR_SCALE = {"us": 1.0, "usecond": 1.0, "ms": 1000.0, "msecond": 1000.0,
              "ns": 0.001, "nsecond": 0.001, "second": 1e6, "s": 1e6}


def load_capture(path: Path) -> dict | None:
    """One CSV -> per-phase, duration-weighted summary. None if unusable."""
    try:
        rows = list(csv.DictReader(path.open()))
    except OSError:
        return None
    if not rows:
        return None

    launches: dict[str, dict] = {}
    for r in rows:
        name = r.get("Metric Name") or ""
        if not name:
            continue
        kid = r["ID"]
        k = launches.setdefault(kid, {"kernel": r.get("Kernel Name", ""), "stalls": {}})
        val = (r.get("Metric Value") or "").replace(",", "")
        try:
            v = float(val)
        except ValueError:
            continue
        if name in SCALARS:
            if name == "Duration":
                v *= _DUR_SCALE.get((r.get("Metric Unit") or "us").strip(), 1.0)
            k[SCALARS[name]] = v
        elif name.startswith(STALL_PREFIX):
            k["stalls"][name[len(STALL_PREFIX):-len(STALL_SUFFIX)]] = v

    launches = {k: v for k, v in launches.items() if "duration" in v}
    if not launches:
        return None

    total = sum(v["duration"] for v in launches.values()) or 1.0

    def wmean(key):
        vals = [(v["duration"], v.get(key)) for v in launches.values()]
        vals = [(d, x) for d, x in vals if x is not None]
        return sum(d * x for d, x in vals) / (sum(d for d, _ in vals) or 1.0)

    stall_names = sorted({s for v in launches.values() for s in v["stalls"]})
    stalls = {}
    for s in stall_names:
        num = sum(v["duration"] * v["stalls"].get(s, 0.0) for v in launches.values())
        stalls[s] = num / total

    sm, mem = wmean("sm_pct"), wmean("mem_pct")
    stall_total = sum(stalls.values()) or 1.0
    top = sorted(stalls.items(), key=lambda kv: -kv[1])

    if max(sm, mem) >= 80.0:
        verdict = "compute bound" if sm >= mem else "memory bound"
    elif sm < 60.0 and mem < 60.0:
        verdict = f"latency bound ({top[0][0]})" if top else "latency bound"
    else:
        verdict = "mixed"

    return {
        "n_kernels": len(launches),
        "total_us": round(total, 2),
        "sm_pct": round(sm, 2),
        "mem_pct": round(mem, 2),
        "occupancy_pct": round(wmean("occupancy_pct"), 2),
        "l2_hit_pct": round(wmean("l2_hit_pct"), 2),
        "verdict": verdict,
        "top_stalls": [{"reason": n, "cycles_per_issue": round(v, 3),
                        "share_of_stall": round(v / stall_total, 3)} for n, v in top[:4]],
        "long_scoreboard": round(stalls.get("long_scoreboard", 0.0), 3),
        "long_scoreboard_share": round(stalls.get("long_scoreboard", 0.0) / stall_total, 3),
    }


def discover(root: str) -> dict:
    """Find every capture as (model, mode, region) -> summary."""
    out: dict[tuple[str, str, str], dict] = {}
    for f in sorted(glob.glob(f"{root}/20260923_E4g_c1_*/ncu/*_details.csv")):
        p = Path(f)
        m = re.match(r"(.+)_(eager|best)_(vision|vlm_prefix|ae_step_0|ae_total)"
                     r"_details\.csv$", p.name)
        if not m:
            continue
        model, mode, region = m.groups()
        s = load_capture(p)
        if s is None:
            continue
        s["csv"] = str(p)
        out[(model, mode, region)] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="results/20260923_E4g_c1_summary")
    ap.add_argument("--md", default="reports/E4g_c1_c2_bottlenecks.md")
    args = ap.parse_args()
    caps = discover(args.root)
    if not caps:
        print("no usable captures yet")
        return 1

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)

    models = sorted({k[0] for k in caps})
    regions = ["vision", "vlm_prefix", "ae_step_0", "ae_total"]

    # ---- C1: what each phase is bound by -------------------------------------------
    rows = []
    print(f"\n{'model':14s}{'mode':6s}{'region':12s}{'SM%':>7s}{'Mem%':>7s}"
          f"{'Occ%':>7s}{'LongSB':>8s}  verdict")
    for model in models:
        for region in regions:
            for mode in ("eager", "best"):
                s = caps.get((model, mode, region))
                if not s:
                    continue
                print(f"{model:14s}{mode:6s}{region:12s}{s['sm_pct']:7.1f}"
                      f"{s['mem_pct']:7.1f}{s['occupancy_pct']:7.1f}"
                      f"{s['long_scoreboard']:8.2f}  {s['verdict']}")
                rows.append({"model": model, "mode": mode, "region": region,
                             **{k: v for k, v in s.items()
                                if k not in ("top_stalls", "csv")},
                             "top_stall": s["top_stalls"][0]["reason"]
                             if s["top_stalls"] else ""})
    with (out / "c1_bounds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- C2: what the best configuration changed, per phase -------------------------
    c2 = []
    print(f"\n{'model':14s}{'region':12s}{'us eager':>10s}{'us best':>10s}"
          f"{'speedup':>9s}{'dSM%':>8s}{'dMem%':>8s}{'dLongSB':>9s}  what changed")
    for model in models:
        for region in regions:
            e, b = caps.get((model, "eager", region)), caps.get((model, "best", region))
            if not (e and b):
                continue
            sp = e["total_us"] / b["total_us"] if b["total_us"] else float("nan")
            d_sm = b["sm_pct"] - e["sm_pct"]
            d_mem = b["mem_pct"] - e["mem_pct"]
            d_lsb = b["long_scoreboard"] - e["long_scoreboard"]
            dk = b["n_kernels"] - e["n_kernels"]
            # Name the mechanism from what moved, rather than assuming compilation
            # "fused" or "sped up" anything.
            bits = []
            if abs(d_lsb) > 0.5:
                bits.append("less memory-latency stall" if d_lsb < 0
                            else "MORE memory-latency stall")
            if d_sm > 5:
                bits.append("higher SM utilisation")
            if d_mem < -5:
                bits.append("less memory pressure")
            if dk:
                bits.append(f"{abs(dk)} {'fewer' if dk < 0 else 'more'} kernels in the "
                            f"first {e['n_kernels']} captured")
            c2.append({"model": model, "region": region,
                       "eager_us": e["total_us"], "best_us": b["total_us"],
                       "speedup": round(sp, 3),
                       "eager_verdict": e["verdict"], "best_verdict": b["verdict"],
                       "d_sm_pct": round(d_sm, 2), "d_mem_pct": round(d_mem, 2),
                       "d_long_scoreboard": round(d_lsb, 3),
                       "mechanism": "; ".join(bits) or "no clear change in the counters"})
            print(f"{model:14s}{region:12s}{e['total_us']:10.1f}{b['total_us']:10.1f}"
                  f"{sp:9.2f}{d_sm:8.1f}{d_mem:8.1f}{d_lsb:9.2f}  {c2[-1]['mechanism']}")
    if c2:
        with (out / "c2_eager_vs_best.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(c2[0].keys()))
            w.writeheader()
            w.writerows(c2)

    # ---- markdown, so the report regenerates itself from the captures ---------------
    if args.md:
        M = []
        M.append("# E4g — what each phase is bound by (C1), and what the best rung fixed (C2)")
        M.append("")
        M.append("**Generated by `bench/analyze_c1.py` from the ncu CSVs.** Nothing is typed "
                 "in. Each capture is one (model, mode, phase): 40 kernel launches with "
                 "SpeedOfLight, MemoryWorkloadAnalysis, WarpStateStats and Occupancy, plus "
                 "the 14 `smsp__average_warps_issue_stalled_*` metrics requested by name.")
        M.append("")
        M.append("Per-kernel numbers are averaged **weighted by kernel duration**; an "
                 "unweighted mean lets a 2 µs elementwise kernel count as much as a 234 µs "
                 "convolution. The bound/unbound thresholds are ncu's own SOLBottleneck "
                 "rule (80 % of SM or memory throughput), not values chosen here: at or "
                 "above 80 % the phase is bound by whichever is higher, below 60 % on both "
                 "it is latency bound and the dominant stall names which latency.")
        M.append("")
        M.append("## C1 — the verdict per phase")
        M.append("")
        M.append("| model | mode | phase | SM % | Mem % | Occ % | LongSB (cyc/issue, "
                 "share) | top stall | verdict |")
        M.append("|---|---|---|---:|---:|---:|---:|---|---|")
        for model in models:
            for region in regions:
                for mode in ("eager", "best"):
                    s_ = caps.get((model, mode, region))
                    if not s_:
                        continue
                    ts = s_["top_stalls"][0] if s_["top_stalls"] else None
                    M.append(
                        f"| {model} | {mode} | `{region}` | {s_['sm_pct']:.1f} | "
                        f"{s_['mem_pct']:.1f} | {s_['occupancy_pct']:.1f} | "
                        f"{s_['long_scoreboard']:.2f} ({100*s_['long_scoreboard_share']:.0f} %) | "
                        f"{ts['reason'] if ts else '—'} "
                        f"({100*ts['share_of_stall']:.0f} %) | {s_['verdict']} |")
        M.append("")
        M.append("## C2 — eager → best, phase by phase")
        M.append("")
        M.append("`us` is the summed duration of the captured kernels in that phase, so it "
                 "compares like with like only within a row; end-to-end phase times are in "
                 "`reports/E4d_phase_balance.md`.")
        M.append("")
        M.append("| model | phase | eager µs | best µs | ratio | eager verdict | "
                 "best verdict | ΔSM % | ΔMem % | ΔLongSB | what moved |")
        M.append("|---|---|---:|---:|---:|---|---|---:|---:|---:|---|")
        for r in c2:
            M.append(f"| {r['model']} | `{r['region']}` | {r['eager_us']:.0f} | "
                     f"{r['best_us']:.0f} | {r['speedup']:.2f}× | {r['eager_verdict']} | "
                     f"{r['best_verdict']} | {r['d_sm_pct']:+.1f} | {r['d_mem_pct']:+.1f} | "
                     f"{r['d_long_scoreboard']:+.2f} | {r['mechanism']} |")
        M.append("")
        M.append(f"Captures summarised: **{len(caps)}**. A phase missing from a table has "
                 "no usable capture yet — `bench/c1_capture_ok.py` rejects any CSV without "
                 "all 14 stall metrics, so a partial capture is never silently averaged in.")
        M.append("")
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(M) + "\n")
        print(f"wrote {args.md}")

    (out / "c1_detail.json").write_text(json.dumps(
        {f"{m}|{mo}|{r}": s for (m, mo, r), s in caps.items()}, indent=1))
    print(f"\nwrote {out}/c1_bounds.csv, c2_eager_vs_best.csv, c1_detail.json")
    print(f"captures summarised: {len(caps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
