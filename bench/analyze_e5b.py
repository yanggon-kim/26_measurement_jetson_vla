"""E5b / T007 — map each kernel's warp stalls onto instruction classes, and say so.

`docs/03_instruction_level_analysis.md` §7 fixes the mapping; this does not invent one.
It is reproduced verbatim in `STALL_CAUSE` below, including the instruction mnemonics,
so a reader can check the attribution against the document rather than against a habit.

Two rules from §7 that this file enforces rather than leaves to the writer:

* **Every stall reason above 5 % must be attributed.** A reason this Orin exports that §7
  does not name — `imc_miss`, `branch_resolving`, `no_instruction`, `membar`, `selected`,
  `sleeping`, `tex_throttle` — is reported under its exported name with "cause not
  documented in §7", never given a guessed cause. `imc_miss` matters here: C1 found the
  eager action expert `imc_miss`-limited in all three LeRobot models, and §7 has no entry
  for it.
* **`selected` / "not selected" is not a problem.** §7: it means the warp was ready and
  another was issued, which is a sign of ample warp-level parallelism. It is excluded
  from the "what is this kernel waiting on" ranking and reported separately, because
  including it makes a healthy kernel look stalled.

Thresholds, also §7: long scoreboard >= 60 % is the signature of a memory-latency-bound
kernel; math pipe throttle > 10 % indicates compute-bound.

**`best` here means C1's `best`** — `--compile-mode reduce-overhead` — and NOT Part B's
best rung (π0 `bf16_both`, π0.5 `bf16_both_graphs`, SmolVLA `vlm_graphs`). The target
kernels were selected from C1's per-kernel data, so the capture has to be of the
configuration those kernels were ranked in. Anything comparing these numbers against
Part B's latencies has to say which "best" it means.

    python bench/analyze_e5b.py --glob 'results/*_E5b_*' --md reports/E5b_summary.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
PIPE_RE = re.compile(r"sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active")

# docs/03_instruction_level_analysis.md §7, verbatim. Do not edit here without editing
# there — the document is the research side's, and this file follows it.
STALL_CAUSE = {
    "long_scoreboard": ("global-memory load (LDG / LD.E) issued; the dependent "
                        "instruction cannot fire until the data returns", "LSU"),
    "short_scoreboard": ("shared-memory load (LDS), constant-cache miss, SHFL, MUFU",
                         "LSU/XU"),
    "mio_throttle": ("LSU issue queue full: too many loads or stores in flight", "LSU"),
    "lg_throttle": ("load/store address calculation, or a shared-memory bank conflict",
                    "LSU"),
    "math_pipe_throttle": ("FMA / tensor-core pipe full: back-pressure from compute",
                           "FMA"),
    "barrier": ("BAR.SYNC / WARPSYNC: waiting for the other warps — load imbalance, or "
                "too few warps to hide it", "control"),
    "wait": ("fixed-latency operation in flight (transcendental, integer divide, FP64), "
             "DEPBAR", "ALU/XU"),
    "selected": ("the warp was READY and another was issued — a sign of ample "
                 "warp-level parallelism, NOT a problem (§7)", "—"),
    "dispatch_stall": ("pipe imbalance — normally small", "control"),
    "drain": ("pipeline drain after a barrier or branch — normally small", "control"),
    "misc": ("pipe imbalance — normally small", "control"),
}
# Exported by this Orin but absent from §7. Named, never guessed at.
UNDOCUMENTED = {"imc_miss", "branch_resolving", "no_instruction", "membar", "sleeping",
                "tex_throttle"}


def load_csv(path: Path) -> dict:
    """{kernel: {stalls, pipes, dur_us, sm_pct}} from an ncu --page details export."""
    ks: dict[str, dict] = {}
    for r in csv.DictReader(path.open()):
        name = (r.get("Kernel Name") or "").strip()
        if not name:
            continue
        k = ks.setdefault(r.get("ID", name), {"name": name, "stalls": {}, "pipes": {}})
        metric = (r.get("Metric Name") or "").strip()
        try:
            v = float((r.get("Metric Value") or "").replace(",", ""))
        except ValueError:
            continue
        if metric.startswith(STALL_PREFIX):
            k["stalls"][metric[len(STALL_PREFIX):].replace(
                "_per_issue_active.ratio", "")] = v
        elif (m := PIPE_RE.fullmatch(metric)):
            k["pipes"][m.group(1)] = v
        elif metric == "Duration":
            unit = (r.get("Metric Unit") or "us").strip()
            k["dur_us"] = v * {"ms": 1e3, "msecond": 1e3, "ns": 1e-3,
                               "nsecond": 1e-3}.get(unit, 1.0)
        elif metric == "Compute (SM) Throughput":
            k["sm_pct"] = v
    return ks


def rows_for(path: Path, model: str, mode: str, region: str) -> list[dict]:
    """One row per (kernel NAME, stall reason), aggregated over the kernel's instances.

    A capture holds many launches of the same kernel. Reporting them separately makes the
    same stall reason appear a dozen times under one heading; reporting only the first
    instance throws most of the evidence away. Both are wrong, so each reason is averaged
    over the instances WEIGHTED BY DURATION -- the same rule C1 uses, so a kernel that
    ran once for 900 us is not outvoted by twelve 3 us launches.
    """
    from crosscheck_sass import _short_name                          # noqa: PLC0415
    merged: dict[str, dict] = {}
    for k in load_csv(path).values():
        if sum(k["stalls"].values()) <= 0:
            continue
        short = _short_name(k["name"]) or k["name"][:40]
        w = max(k.get("dur_us", 0.0), 1e-9)
        m = merged.setdefault(short, {"stalls": {}, "pipes": {}, "dur": 0.0,
                                      "sm": 0.0, "n": 0})
        for reason, v in k["stalls"].items():
            m["stalls"][reason] = m["stalls"].get(reason, 0.0) + v * w
        for pipe, v in k["pipes"].items():
            m["pipes"][pipe] = m["pipes"].get(pipe, 0.0) + v * w
        m["sm"] += k.get("sm_pct", 0.0) * w
        m["dur"] += w
        m["n"] += 1

    out = []
    for short, mg in merged.items():
        w = mg["dur"]
        k = {"stalls": {r: v / w for r, v in mg["stalls"].items()},
             "pipes": {p: v / w for p, v in mg["pipes"].items()},
             "dur_us": mg["dur"], "sm_pct": mg["sm"] / w}
        total = sum(k["stalls"].values())
        if total <= 0:
            continue
        # §7: `selected` is warps that were READY. Excluding it from the denominator is
        # what makes the remaining shares mean "of the time this kernel was WAITING".
        waiting = total - k["stalls"].get("selected", 0.0)
        for reason, v in sorted(k["stalls"].items(), key=lambda kv: -kv[1]):
            share = 100 * v / total
            # `selected` is not waiting, so it gets no share OF waiting -- printing one
            # makes the column sum past 100 % and invites reading it as a stall to fix,
            # which §7 says explicitly it is not.
            wshare = ("" if reason == "selected"
                      else round(100 * v / waiting, 2) if waiting > 0 else "")
            cause, cls = STALL_CAUSE.get(
                reason, ("cause not documented in docs/03 §7 — reported under its "
                         "exported name rather than given a guessed cause", "?"))
            out.append({
                "model": model, "mode": mode, "region": region, "kernel": short,
                "duration_us": round(k["dur_us"], 3),
                "launches": mg["n"],
                "sm_throughput_pct": round(k["sm_pct"], 2),
                # docs/03 §1 excludes a kernel under 5 % SM throughput whose duration is
                # comparable to the launch gap: it is launch-bound and its instruction mix
                # explains nothing about why the phase is slow.
                #
                # Such kernels are NOT targets, but they still appear here, because the ncu
                # capture matches a regex on the short name and several short names are
                # substrings of one another -- the capture named `elementwise_kernel` for
                # pi0/eager holds 14 distinct kernels, among them vectorized_ and
                # unrolled_elementwise_kernel and elementwise_kernel_with_index. The
                # per-kernel attribution is still correct, since rows merge by _short_name,
                # but without this flag a launch-bound kernel's stall mix sits in the report
                # looking like an explanation. Marked rather than dropped: the data is real
                # and someone may want it, they just must not reason from it.
                "launch_bound_s1": ("yes" if (k["sm_pct"] < 5.0 and
                                              k["dur_us"] / max(mg["n"], 1) < 62.0)
                                    else ""),
                "stall_reason": reason,
                "share_of_all_stalls_pct": round(share, 2),
                "share_of_waiting_pct": wshare,
                "above_5pct": "yes" if share > 5 else "",
                "instruction_class": cls,
                "documented_in_s7": "no" if reason in UNDOCUMENTED else "yes",
                "cause": cause,
                "pipe_lsu_pct": round(k["pipes"].get("lsu", float("nan")), 2),
                "pipe_fma_pct": round(k["pipes"].get("fma", float("nan")), 2),
                "pipe_alu_pct": round(k["pipes"].get("alu", float("nan")), 2),
            })
    return out


def verdict(rows: list[dict]) -> str:
    """§7's own thresholds, applied rather than paraphrased."""
    by = {r["stall_reason"]: r["share_of_waiting_pct"] for r in rows
          if isinstance(r["share_of_waiting_pct"], float)}
    ls = by.get("long_scoreboard", 0.0)
    mp = by.get("math_pipe_throttle", 0.0)
    if ls >= 60:
        return f"memory-latency bound (long scoreboard {ls:.1f} % >= 60 %, §7)"
    if mp > 10:
        return f"compute bound (math pipe throttle {mp:.1f} % > 10 %, §7)"
    top = max(((v, k) for k, v in by.items() if k != "selected"), default=(0, "—"))
    return f"neither threshold met; largest wait is {top[1]} at {top[0]:.1f} %"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*_E5b_*")
    ap.add_argument("--md", default="reports/E5b_summary.md")
    args = ap.parse_args()

    all_rows: list[dict] = []
    per_dir: dict[str, list[dict]] = {}
    for d in sorted(glob.glob(args.glob)):
        for f in sorted(Path(d).glob("ncu/*_details.csv")):
            # <model>_<mode>_<region>_<kernel>_details.csv. The kernel is part of the
            # tag because all targets of one (model, mode) share a region; without it the
            # captures overwrite each other.
            m = re.match(r"(.+?)_(eager|best)_(ae_step_0|ae_step|ae_total|vision|vlm_prefix)"
                         r"(?:_.+)?_details\.csv$", f.name)
            if not m:
                continue
            rs = rows_for(f, *m.groups())
            per_dir.setdefault(d, []).extend(rs)
            all_rows += rs
    if not all_rows:
        print(f"no E5b captures under {args.glob} — nothing to analyse")
        return 1

    # One CSV per RUN directory, holding that run's rows. The first version wrote every
    # row into whichever directory sorted last, which is arbitrary: five runs would have
    # had no stall_attribution.csv and the sixth would have held all six runs' rows under
    # its own name.
    fields = list(all_rows[0].keys())
    for d, rs in sorted(per_dir.items()):
        out = Path(d) / "stall_attribution.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rs)
        print(f"wrote {out} ({len(rs)} rows)")
    # and one combined file beside the report, for the cross-model comparison
    comb = Path(args.md).with_suffix(".csv")
    comb.parent.mkdir(parents=True, exist_ok=True)
    with comb.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {comb} ({len(all_rows)} rows, all runs)")

    L = ["# E5b — which instructions the action-expert kernels wait on", "",
         "**Generated by `bench/analyze_e5b.py` from the committed ncu CSVs.** The stall "
         "-> instruction-class mapping is `docs/03_instruction_level_analysis.md` §7, "
         "reproduced verbatim in the script; nothing here invents a cause.", "",
         "Two rules from §7 that the tables below follow:", "",
         "* **`selected` is not a stall to fix.** It means the warp was ready and another "
         "was issued — ample warp-level parallelism. It is excluded from the "
         "\"waiting\" denominator and shown separately, because counting it makes a "
         "healthy kernel look stalled.",
         "* **A reason §7 does not name is reported under its exported name**, with its "
         "cause left open. `imc_miss` is the one that matters here: C1 found the eager "
         "action expert `imc_miss`-limited in all three LeRobot models, and §7 has no "
         "entry for it.", "",
         "Thresholds are §7's: long scoreboard >= 60 % means memory-latency bound; math "
         "pipe throttle > 10 % means compute bound.", ""]

    groups: dict[tuple, list[dict]] = {}
    for r in all_rows:
        groups.setdefault((r["model"], r["mode"], r["region"], r["kernel"]), []).append(r)
    for (model, mode, region, kernel), rows in sorted(groups.items()):
        rows.sort(key=lambda r: -r["share_of_all_stalls_pct"])
        # §1's verdict, carried into the heading. These kernels are in the capture only
        # because the ncu regex matches short names that are substrings of one another, and
        # without this marker a launch-bound kernel's stall table reads as an explanation
        # when §1 says its instruction mix explains nothing about why the phase is slow.
        _lb = rows[0].get("launch_bound_s1") == "yes"
        _mark = " — ⚠ LAUNCH-BOUND (§1): not a target; its stall mix explains nothing" if _lb else ""
        L += [f"### {model} / {mode} / {region} — `{kernel}`{_mark}", "",
              f"{rows[0]['duration_us']:.1f} µs, SM throughput "
              f"{rows[0]['sm_throughput_pct']:.1f} %. **{verdict(rows)}**"
              + ("  \n*Swept into the capture by a substring match on the kernel name. "
                 "Under 5 % SM throughput and shorter per launch than the 62 µs launch gap, "
                 "so §1 excludes it from the instruction-level procedure. Shown for "
                 "completeness; do not reason from it.*" if _lb else ""), "",
              "| stall reason | % of waiting | % of all | class | in §7 | cause |",
              "|---|---:|---:|---|---|---|"]
        for r in rows:
            if r["share_of_all_stalls_pct"] < 1 and not r["above_5pct"]:
                continue
            ws = r["share_of_waiting_pct"]
            L.append(f"| `{r['stall_reason']}` | "
                     + (f"{ws:.1f}" if isinstance(ws, float) else "— (ready, not waiting)")
                     + " | "
                     f"{r['share_of_all_stalls_pct']:.1f} | {r['instruction_class']} | "
                     f"{r['documented_in_s7']} | {r['cause']} |")
        L.append("")
        missing = [r["stall_reason"] for r in rows
                   if r["above_5pct"] and r["documented_in_s7"] == "no"]
        if missing:
            L += [f"**Above 5 % and not documented in §7: {', '.join(missing)}.** "
                  f"Reported under the exported name; the cause is left open rather than "
                  f"guessed.", ""]

    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(L) + "\n")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
