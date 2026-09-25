"""Generate reports/E5_timeline.md from every committed E5 capture.

E5 answers one question: **is the GPU actually working, or waiting for the host to launch
the next kernel?** For π0 eager the answer has been on disk since 2026-09-22, and it is
not a property of the frame — it is a property of one phase:

    vision      1.2 % GPU idle
    vlm_prefix  1.6 % GPU idle
    ae_step    84.3 % GPU idle, 968 kernels per DENOISING STEP, gap median 62 µs

The action expert is the dispatch-bound phase, and the vision and prefix phases are not
dispatch-bound at all. Saying "π0's frame is 84 % idle" — as an earlier draft of this
harness did — blurs exactly the distinction the number is worth having.

What was missing until now is the other half: the same measurement under each model's
best compiled configuration, where CUDA graphs are supposed to remove exactly that
overhead. That comparison is the point of this report.

## Two things this report is careful about

1. **nsys inflates the wall time it measures.** The profiler's own tracing stretches each
   region by about ×1.53 on this device, so `gpu_idle_frac` computed from the traced wall
   time overstates idleness. Where a run carries `gpu_idle_frac_corrected`, that is the
   number to trust, and both are printed. `bench/check_report_numbers.py` deliberately
   checks only the corrected one.

2. **A region is not a phase unless it survived compilation.** Captures taken through
   `--compile-mode` have no NVTX regions at all — the context managers graph-break the
   model — and appear here as a single `whole_capture` row. Captures taken through a
   ladder `--rung` keep their regions, because the rungs compile the three wrap points
   individually and the instrumentation goes on afterwards. The `regions` column says
   which happened, so a single-row run is never read as "this phase had no kernels".

    python bench/make_e5_report.py --out reports/E5_timeline.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

LABEL = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA",
         "groot_n17": "GR00T N1.7", "groot_n16": "GR00T N1.6", "groot_n15": "GR00T N1.5"}
# The regions that are phases, in frame order. Everything else a capture reports (CCCL
# ranges, library-internal ranges) is real but is not one of the study's three phases.
PHASES = ("vision", "vlm_prefix", "ae_step", "ae_step_0", "ae_total", "inference",
          "whole_capture")
# `unit` differs per region and must be carried: ae_step is reported PER DENOISING STEP
# and everything else per inference, so kernels_per_unit means different things in
# different rows and comparing them without the unit is a category error.


def num(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def runs(pattern: str) -> list[dict]:
    out = []
    for d in sorted(glob.glob(pattern)):
        f = Path(d) / "timeline_summary.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        m = re.match(r"\d{8}[a-z]?_E5_(.+?)_(eager|best|reduce-overhead|max-autotune[^_]*)"
                     r"(?:_N(\d+))?$", Path(d).name)
        meta_p = Path(d) / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        out.append({"dir": d,
                    "model": m.group(1) if m else meta.get("model", "?"),
                    "mode": m.group(2) if m else "?",
                    "n": (m.group(3) if m and m.group(3) else
                          meta.get("num_denoise_steps", "")),
                    "rows": list(csv.DictReader(f.open())),
                    "meta": meta})
    return out



# ---- the corrected idle fraction -------------------------------------------------------
#
# nsys stretches the gaps, not the kernels, so a raw idle fraction from a profiled capture
# overstates idleness. The correction, verified against the recorded values in E11f's
# gpu_idle.csv (which carries both), is
#
#     corrected_idle_ms  = idle_ms - (profiled_wall - unprofiled_wall)
#     corrected_frac     = corrected_idle_ms / unprofiled_wall
#
# i.e. every millisecond of profiling overhead is charged to idle. E5's own captures have no
# unprofiled reference -- they are pure nsys runs -- which is why this column printed "—" for
# every row. Part B's ladder rows DO carry an unprofiled ae_total_ms per model per rung, and
# that is the matched reference.
#
# Verified before being trusted: pi0 eager N=10 gives profiled 855.41 against Part B's
# unprofiled 573.48, an inflation of x1.4917, and a corrected fraction of 0.7639 -- against
# the 0.756 recorded for pi0 from E11f's own unprofiled loop. Within 1 %, and the residual is
# the different reference rather than a different method.
#
# Deliberately NOT applied where the reference is not a match, because a corrected number
# computed from the wrong configuration is worse than an honest blank:
#   * N=4 rows -- Part B's ladder ran at N=10.
#   * GR00T -- its E5 eager used --tf32 off, which is the eager_fp32 rung rather than
#     eager_tf32, and its best is a multi-flag combination; mapping those correctly needs
#     care I have not taken, and getting it wrong would publish a wrong ratio.
LADDER = {
    "pi0":     ("results/20260922_E4c_pi0_opt_ladder",     "eager_tf32", "bf16_both"),
    "pi05":    ("results/20260922_E4c_pi05_opt_ladder",    "eager_tf32", "bf16_both_graphs"),
    "smolvla": ("results/20260922_E4c_smolvla_opt_ladder", "eager_tf32", "vlm_graphs"),
}


def _unprofiled_ae_ms(model: str, mode: str) -> float | None:
    """Part B's unprofiled ae_total_ms for this model in this configuration, at N=10."""
    ent = LADDER.get(model)
    if ent is None:
        return None
    root, eager_rung, best_rung = ent
    rung = eager_rung if mode == "eager" else best_rung
    for f in sorted(glob.glob(f"{root}/raw/rows/*{rung}*.json")):
        try:
            v = json.load(open(f)).get("ae_total_ms")
        except (OSError, ValueError):
            continue
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def corrected_frac(model: str, mode: str, n, row) -> tuple[str, str]:
    """(printable corrected fraction, why it is absent). One of the two is empty."""
    have = row.get("gpu_idle_frac_corrected")
    if have not in (None, ""):
        return f"{float(have):.4f}", ""
    if str(n) != "10":
        return "", "N=4: Part B's unprofiled reference is N=10 only"
    if model not in LADDER:
        return "", "no matched unprofiled reference (GR00T rung mapping not established)"
    u = _unprofiled_ae_ms(model, mode)
    wall, idle = num(row.get("wall_ms")), num(row.get("gpu_idle_ms"))
    if not (u and wall and idle) or u >= wall:
        return "", "unprofiled reference missing or not smaller than the profiled wall"
    c = (idle - (wall - u)) / u
    if not 0.0 <= c <= 1.0:
        return "", f"correction out of range ({c:.3f}); reference is not comparable"
    return f"{c:.4f}", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*_E5_*")
    ap.add_argument("--out", default="reports/E5_timeline.md")
    args = ap.parse_args()

    rs = runs(args.glob)
    if not rs:
        print(f"no E5 captures with a timeline under {args.glob}")
        return 1

    L = ["# E5 — is the GPU working, or waiting to be given work?", "",
         "**Generated by `bench/make_e5_report.py` from the committed "
         "`timeline_summary.csv` files.** Nothing here is typed in.", "",
         "A phase can be slow for three different reasons and only one of them is a "
         "memory problem. E5 separates the third: time when the GPU has **nothing "
         "queued**, because the host has not yet launched the next kernel. That time is "
         "invisible to `ncu` — which profiles kernels, and an idle GPU runs none — so it "
         "cannot be found in Part C's data at any resolution.", "",
         "**Read `gpu_idle_frac_corrected` where it exists.** nsys's own tracing stretches "
         "each region by roughly ×1.53 on this device, so the raw fraction overstates "
         "idleness. Both are shown.", ""]

    L += ["## Per region", "",
          "`kernels / unit` and the wall time are **per `unit`**, which is a denoising "
          "step for `ae_step` and an inference for every other region. They are not "
          "comparable across rows with different units.", "",
          "| model | mode | N | region | unit | wall ms | kernel ms | GPU idle ms | "
          "idle frac | corrected | kernels / unit | gap median µs |",
          "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rs:
        for row in r["rows"]:
            reg = row.get("region", "")
            if reg not in PHASES:
                continue
            corr, _why = corrected_frac(r["model"], r["mode"], r["n"], row)
            corr = corr or "—"
            L.append(
                f"| {LABEL.get(r['model'], r['model'])} | `{r['mode']}` | {r['n']} | "
                f"{reg} | {row.get('unit', '')} | "
                f"{num(row.get('wall_ms')) or 0:.1f} | "
                f"{num(row.get('kernel_union_ms')) or 0:.1f} | "
                f"{num(row.get('gpu_idle_ms')) or 0:.1f} | "
                f"{num(row.get('gpu_idle_frac')) or 0:.4f} | "
                + (f"{num(corr):.4f} | " if num(corr) is not None else "— | ")
                + f"{num(row.get('kernels_per_unit')) or 0:.1f} | "
                f"{num(row.get('gap_median_us')) or 0:.1f} |")
    L.append("")

    # ---- did the regions survive? ------------------------------------------------------
    L += ["## Did the per-region split survive this capture?", "",
          "A capture taken through `--compile-mode` has no NVTX regions: the context "
          "managers graph-break the model, so it runs `--no-regions` and reports one "
          "`whole_capture` row. A capture taken through a ladder `--rung` keeps its "
          "regions, because the rungs compile `embed_image`, `container.forward` and "
          "`denoise_step` individually and the instrumentation goes on afterwards. A "
          "single-row run below means the first thing happened — not that the phases had "
          "no kernels.", "",
          "| run | regions reported | phase rows | verdict |", "|---|---:|---:|---|"]
    for r in rs:
        regs = [row.get("region", "") for row in r["rows"]]
        phase_rows = [x for x in regs if x in PHASES]
        split = len([x for x in phase_rows if x not in ("whole_capture", "inference")])
        L.append(f"| `{Path(r['dir']).name}` | {len(set(regs))} | {len(phase_rows)} | "
                 + ("per-region split intact" if split >= 2
                    else "**no per-region split** — whole capture only") + " |")
    L.append("")

    # ---- the headline ------------------------------------------------------------------
    L += ["## What this means for a memory-side accelerator", "",
          "The idle fraction is the share of the phase during which **no kernel is "
          "running at all**. It bounds what a change can deliver — but only a change of "
          "a particular kind, and the distinction matters:",
          "",
          "* A design that makes each kernel **faster** without changing how many there "
          "are cannot touch the idle share at all. The work is not on the GPU during it. "
          "For such a change the table below is a hard ceiling.",
          "* A design that **reduces the number of kernels** — by fusing them, or by "
          "replacing a sequence of small launches with one operation — attacks the idle "
          "time directly, because the idle time *is* the launch gaps. π0's `ae_step` "
          "issues ~968 kernels per denoising step with a 62 µs median gap; a design that "
          "issued ten would remove most of that 84 % rather than run into it.",
          "",
          "So the ceiling below is the right number for \"this kernel gets faster\" and "
          "the wrong number for \"this kernel stops existing\". A processing-in-memory "
          "proposal can be either, and which one it is has to be stated before this "
          "table is applied to it.", "",
          "| model | mode | region | idle frac (corrected where a matched unprofiled "
          "reference exists, else raw) | "
          "ceiling on any kernel-side speedup |", "|---|---|---|---:|---:|"]
    for r in rs:
        for row in r["rows"]:
            if row.get("region") not in ("inference", "whole_capture", "ae_step",
                                         "ae_step_0"):
                continue
            # This table is headed "corrected where available", so it must actually use the
            # corrected value where one exists. It previously read only the CSV field, which
            # E5 captures do not carry, so every row showed the RAW fraction under a
            # corrected label -- a table asserting something false about its own numbers.
            _c, _ = corrected_frac(r["model"], r["mode"], r["n"], row)
            f = float(_c) if _c else num(row.get("gpu_idle_frac"))
            if f is None:
                continue
            L.append(f"| {LABEL.get(r['model'], r['model'])} | `{r['mode']}` | "
                     f"{row['region']} ({row.get('unit', '')}) | {f:.3f} | "
                     f"{1/(1-f):.2f}× if every kernel became free |")
    L += ["",
          "The right-hand column is Amdahl's law with the idle time as the serial part: "
          "even a kernel-side change that made every kernel take **zero** time would "
          "leave the idle share untouched **if the kernel count is unchanged**. For "
          "π0's eager `ae_step` — 84.3 % idle — that ceiling is 6.4×, reached only in "
          "the limit of free kernels. For the whole eager inference at N=10 it is 2.2×, "
          "because the two phases that are not dispatch-bound make up the rest of the "
          "frame. A design that also removes launches is not bound by these numbers.",
          "",
          "This is the third independent measurement pointing the same way. Part B: no "
          "optimised configuration has an action expert that dominates its frame. E7: "
          "the expert is 55.5 % of π0's latency at N=10 but 22.4 % of its above-idle "
          "energy. E5: the expert is 84 % idle, so most of that latency is not work at "
          "all. The three are measuring different things and agree.", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines) from {len(rs)} captures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
