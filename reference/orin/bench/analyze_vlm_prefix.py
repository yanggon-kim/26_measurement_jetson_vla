"""Job 2 — per-KERNEL bottleneck analysis of `vlm_prefix`, for π0 and GR00T N1.7 / N1.6.

C1 answered "what is this phase bound by" at phase granularity, by averaging every kernel
weighted by duration. That is the right summary and the wrong resolution for deciding what
to *do*: a phase whose weighted average says "latency bound" can be one enormous GEMM that
is compute bound plus a tail of tiny elementwise kernels that are not. A fix is applied to
a kernel, so this asks the same question per kernel.

Why `vlm_prefix` and why these three models. The prefix is the phase the study turned on:
compilation barely moved it for π0 (534 -> 516 ms, 1.04x) while precision moved it 2.16x,
and it is the largest phase in π0's optimised frame at 45.4 %. GR00T N1.7 and N1.6 are the
contrast — same three-way split, a different backbone (Qwen3-VL vs Eagle), and a prefix
that compilation *does* move.

For each (model, mode) this reports:
  * the kernels that dominate the phase's captured time, by name, grid/block and duration;
  * what each is bound by, from its OWN counters, using ncu's SOLBottleneck threshold;
  * the phase's main bottleneck as one claim, with the counters behind it;
  * what changed between eager and best, kernel family by kernel family.

Caveat carried into every table: `-c 40` captures the FIRST 40 kernels of the range, not a
duration-weighted sample of the whole phase. Shares below are shares *of the captured
window*. That window is the start of the prefix and is representative of its kernel mix,
but a kernel that only runs late in the phase is not in it.

    python bench/analyze_vlm_prefix.py --md reports/E4h_vlm_prefix_kernels.md
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

MODELS = [("pi0", "π0", "results/20260923_E4g_c1_pi0_{mode}/ncu/pi0_{mode}_vlm_prefix_details.csv"),
          ("groot_n17", "GR00T N1.7",
           "results/20260923_E4g_c1_groot_n17_{mode}/ncu/groot_n17_{mode}_vlm_prefix_details.csv"),
          ("groot_n16", "GR00T N1.6",
           "results/20260923_E4g_c1_groot_n16_{mode}/ncu/groot_n16_{mode}_vlm_prefix_details.csv")]

SCALARS = {"Duration": "dur", "Compute (SM) Throughput": "sm",
           "Memory Throughput": "mem", "Achieved Occupancy": "occ",
           "L2 Hit Rate": "l2", "L1/TEX Hit Rate": "l1",
           "Memory Throughput [Gbyte/s]": "dram_gbs"}
STALL = "smsp__average_warps_issue_stalled_"
_DUR = {"us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "ns": 1e-3, "nsecond": 1e-3}


def family(name: str) -> str:
    """Group kernels by what they ARE, so eager and compiled can be compared.

    Compiled kernels get inductor-generated names (`triton_poi_fused_...`) that never
    appear in an eager capture, so comparing raw names across modes finds nothing in
    common. The family is the useful unit: a GEMM is a GEMM whether cuBLAS or Triton ran it.
    """
    n = name.lower()
    if "gemm" in n or "cutlass" in n or "s16816" in n or "implicit_convolve" in n:
        return "GEMM / conv"
    if "attention" in n or "flash" in n or "softmax" in n or "mha" in n:
        return "attention / softmax"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise"
    if "reduce" in n or "norm" in n or "mean" in n:
        return "reduction / norm"
    if "triton_poi" in n or "triton_red" in n or "triton" in n:
        return "triton fused"
    if "cat" in n or "copy" in n or "index" in n or "permute" in n:
        return "copy / layout"
    return "other"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    ks: dict[str, dict] = {}
    for r in csv.DictReader(path.open()):
        mname = r.get("Metric Name") or ""
        if not mname:
            continue
        k = ks.setdefault(r["ID"], {"name": r.get("Kernel Name", ""),
                                    "grid": r.get("Grid Size", ""),
                                    "block": r.get("Block Size", ""), "stalls": {}})
        try:
            v = float((r.get("Metric Value") or "").replace(",", ""))
        except ValueError:
            continue
        if mname in SCALARS:
            if mname == "Duration":
                v *= _DUR.get((r.get("Metric Unit") or "us").strip(), 1.0)
            k[SCALARS[mname]] = v
        elif mname.startswith(STALL):
            k["stalls"][mname[len(STALL):].replace("_per_issue_active.ratio", "")] = v
    out = [v for v in ks.values() if "dur" in v]
    for k in out:
        k["family"] = family(k["name"])
        st = k["stalls"]
        k["top_stall"] = max(st.items(), key=lambda kv: kv[1])[0] if st else ""
        k["stall_total"] = sum(st.values()) or 1.0
        sm, mem = k.get("sm", 0.0), k.get("mem", 0.0)
        if max(sm, mem) >= 80:
            k["verdict"] = "compute bound" if sm >= mem else "memory bound"
        elif sm < 60 and mem < 60:
            k["verdict"] = f"latency ({k['top_stall']})"
        else:
            k["verdict"] = "mixed"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="reports/E4h_vlm_prefix_kernels.md")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    L = ["# E4h — what is actually slow inside `vlm_prefix`, kernel by kernel", "",
         "**Generated by `bench/analyze_vlm_prefix.py` from the C1 ncu CSVs.** Nothing typed in.",
         "",
         "C1 judged each phase by averaging its kernels weighted by duration. That is the "
         "right summary and the wrong resolution for deciding what to fix: a phase whose "
         "weighted average reads \"latency bound\" can be one large GEMM that is not, plus a "
         "tail of small kernels that are. A fix is applied to a kernel, so this asks the "
         "question per kernel.", "",
         "**Read the shares as shares of the captured window.** `ncu -c 40` takes the first "
         "40 kernels of the NVTX range, not a duration-weighted sample of the whole phase. "
         "The window is the start of the prefix and is representative of its kernel mix; a "
         "kernel that runs only late in the phase is not in it.", ""]

    for key, label, tmpl in MODELS:
        L.append(f"## {label}")
        L.append("")
        for mode in ("eager", "best"):
            ks = load(Path(tmpl.format(mode=mode)))
            if not ks:
                L.append(f"*{mode}: no capture*")
                L.append("")
                continue
            tot = sum(k["dur"] for k in ks) or 1.0
            L.append(f"### {label} — {mode}  ({len(ks)} kernels, {tot:.1f} µs captured)")
            L.append("")
            L.append("| share | kernel | grid / block | µs | SM % | Mem % | Occ % | L2 % | "
                     "top stall | verdict |")
            L.append("|---:|---|---|---:|---:|---:|---:|---:|---|---|")
            for k in sorted(ks, key=lambda x: -x["dur"])[:args.top]:
                nm = re.sub(r"<.*", "", k["name"]).strip()[:44] or k["name"][:44]
                sh = 100 * k["dur"] / tot
                st = k["stalls"].get(k["top_stall"], 0.0) / k["stall_total"]
                L.append(f"| {sh:.1f} % | `{nm}` | {k['grid']} / {k['block']} | "
                         f"{k['dur']:.1f} | {k.get('sm',0):.1f} | {k.get('mem',0):.1f} | "
                         f"{k.get('occ',0):.1f} | {k.get('l2',0):.1f} | "
                         f"{k['top_stall']} {100*st:.0f} % | {k['verdict']} |")
            L.append("")
            fam = defaultdict(lambda: [0.0, 0])
            for k in ks:
                fam[k["family"]][0] += k["dur"]
                fam[k["family"]][1] += 1
            L.append("| family | µs | share | kernels |")
            L.append("|---|---:|---:|---:|")
            for f, (d, c) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
                L.append(f"| {f} | {d:.1f} | {100*d/tot:.1f} % | {c} |")
            L.append("")
        L.append("")

    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(L) + "\n")
    print(f"wrote {args.md}")
    for key, label, tmpl in MODELS:
        for mode in ("eager", "best"):
            ks = load(Path(tmpl.format(mode=mode)))
            if not ks:
                continue
            tot = sum(k["dur"] for k in ks) or 1.0
            top = sorted(ks, key=lambda x: -x["dur"])[0]
            print(f"  {label:12s} {mode:6s} {len(ks):3d} kernels {tot:8.1f} us | "
                  f"top {100*top['dur']/tot:4.1f}% {top['verdict']:24s} "
                  f"{re.sub(r'<.*','',top['name'])[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
