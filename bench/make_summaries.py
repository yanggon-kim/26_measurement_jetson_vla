"""Protocol §5 `summary.md` for result directories that lack one (2026-09-28 completeness audit).

§5: "summary.md: five to fifteen lines — what was run, the headline table, anything odd." 277 result directories were
committed without it. This writes one per directory from that directory's OWN files — meta.json and its primary CSV —
and never recomputes or invents a number: the headline table is the first rows of the primary CSV as committed.
Directories that already have summary.md are left untouched. Each file says it was generated and from what.

    python3 bench/make_summaries.py [--dry-run] [results/<dir> ...]   (named dirs only -- never a run still writing)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

PRIMARY = ("latency_summary.csv", "summary.csv", "stages.csv", "corun_ratios.csv", "power.csv", "sigma.csv",
           "dram_bytes.csv", "verdicts.csv", "roofline.csv", "floor.csv", "timeline_summary.csv", "opt_ladder.csv",
           "phase_split.csv", "sm_sensitivity.csv", "equivalence.csv", "episodes.csv", "model_inputs.csv",
           "kernel_bytes.csv", "gemm_excess_by_shape.csv", "kv_handover.csv", "latency_samples.csv")
ODD_KEYS = ("note", "deviation", "anomal", "warning", "invalid", "caveat", "correction", "backfill", "not_measured")


def first_csv(d):
    for name in PRIMARY:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    cs = [x for x in sorted(glob.glob(os.path.join(d, "*.csv"))) + sorted(glob.glob(os.path.join(d, "*", "*.csv")))
          if os.path.isfile(x)]
    return cs[0] if cs else None


def table(p, max_rows=6, max_cols=8, width=24):
    with open(p, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], 0
    h = rows[0][:max_cols]
    body = [r[:max_cols] for r in rows[1:1 + max_rows]]
    cut = lambda x: (x if len(x) <= width else x[:width - 1] + "…").replace("|", "/")  # noqa: E731
    out = ["| " + " | ".join(cut(x) for x in h) + " |", "|" + "---|" * len(h)]
    out += ["| " + " | ".join(cut(x) for x in r) + " |" for r in body]
    return out, len(rows) - 1


def main() -> int:
    dry = "--dry-run" in sys.argv
    n = 0
    named = [a for a in sys.argv[1:] if not a.startswith("--")]
    for d in (named or sorted(glob.glob("results/*/"))):
        d = d.rstrip("/")
        if os.path.exists(os.path.join(d, "summary.md")):
            continue
        metas = [os.path.join(d, "meta.json")] if os.path.exists(os.path.join(d, "meta.json")) else \
            sorted(glob.glob(os.path.join(d, "*", "meta.json")))
        m = {}
        if metas:
            try:
                m = json.load(open(metas[0]))
            except Exception:  # noqa: BLE001
                m = {}
        run = m.get("run") if isinstance(m.get("run"), dict) else {}
        exp = m.get("experiment") or run.get("experiment")
        lines = [f"# {os.path.basename(d)}", ""]
        lines.append(f"* **What:** {exp or 'see meta.json'}" + (f" — task {m['task']}" if m.get("task") else ""))
        gate = m.get("gate") if isinstance(m.get("gate"), dict) else {}
        pm = m.get("power_mode") or gate.get("power_mode")
        bits = [f"date {m.get('date_utc')}" if m.get("date_utc") else "", f"power mode {pm}" if pm else "",
                f"model {m.get('model')}" if m.get("model") else "", f"rung / mode {m.get('rung') or m.get('mode')}" if (m.get("rung") or m.get("mode")) else ""]
        bits = [b for b in bits if b]
        if bits:
            lines.append("* **Conditions:** " + ", ".join(str(b) for b in bits))
        if m.get("command"):
            lines.append(f"* **Command:** `{str(m['command'])[:220]}`")
        if len(metas) > 1:
            lines.append(f"* **Sub-runs:** {', '.join(os.path.basename(os.path.dirname(x)) for x in metas[:8])}"
                         + (" …" if len(metas) > 8 else ""))
        p = first_csv(d)
        if p:
            t, nrows = table(p)
            lines += ["", f"Headline — first rows of `{os.path.relpath(p, d)}` ({nrows} rows):", ""] + t
        odd = []
        for k, v in m.items():
            if any(o in k.lower() for o in ODD_KEYS) and v not in (None, "", [], {}):
                odd.append(f"{k}: {str(v)[:200]}")
        for k in ("n_nonfinite", "nonfinite", "valid", "throttled"):
            if k in m and m[k] not in (None, "", 0, False, "False") and not (k == "valid" and m[k] is True):
                odd.append(f"{k}: {m[k]}")
        lines += ["", "**Anything odd:** " + ("; ".join(odd[:3]) if odd else "nothing flagged in meta.json.")]
        lines += ["", "*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV "
                      "(protocol §5 completeness audit); no number was recomputed. The report that analyses this run is "
                      "the authority.*"]
        lines = [x for x in lines]
        if not dry:
            open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
        n += 1
    print(f"{'would write' if dry else 'wrote'} {n} summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
