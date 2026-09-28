"""Per-model reports that the task files name, extracted VERBATIM from the consolidated ones.

The measurements for T004 / T007 / T008 / T009 / T011 / T012 were reported in cross-model
files (reports/E5_timeline.md, E7_power_energy.md, E5b_summary.md, E11_models.md), while each
task's acceptance names per-model files (reports/E5_pi05_timeline.md, ...). Watchdog §1e
therefore shows those tasks with named deliverables missing, three of them marked `done`.

This writes each named file as an EXTRACT, not a re-derivation:
  * every markdown table keeps its header and ONLY the rows naming this model;
  * every `###` subsection is kept only if its heading names this model;
  * prose is NOT copied, because a paragraph in a cross-model report usually argues about
    several models at once and would read as a claim about this one. The consolidated
    report is linked for it.
Nothing is recomputed, so an extract can never disagree with its source. Whether an extract
satisfies acceptance is the research side's call and is asked in STATUS.md.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MATCH = {
    "pi0": re.compile(r"(π0(?![.\d])|pi0(?![5\d]))"),
    "pi05": re.compile(r"(π0\.5|pi05)"),
    "smolvla": re.compile(r"(?i)smolvla"),
    "groot": re.compile(r"(GR00T|groot)"),
    "groot_n17": re.compile(r"(GR00T N1\.7|groot_n17|N1\.7)"),
}
LABEL = {"pi0": "π0", "pi05": "π0.5", "smolvla": "SmolVLA", "groot": "GR00T (N1.5, N1.6, N1.7)",
         "groot_n17": "GR00T N1.7"}

# (output file, task, source report, model key, what the task asks the file to contain)
# 2026-09-28: reports/E5b_{pi0,pi05,smolvla}.md are no longer extracts -- they follow docs/03 §10's seven parts and
# are written by bench/make_e5b_reports.py, so they are NOT in this list (re-running this script cannot overwrite them).
TARGETS = [
    ("E5_pi0_timeline.md", "T004", "E5_timeline.md", "pi0", "kernels executing vs GPU waiting, per mode"),
    ("E7_pi0_power.md", "T004", "E7_power_energy.md", "pi0", "W per rail and phase, energy per inference and per step"),
    ("E5_pi05_timeline.md", "T008", "E5_timeline.md", "pi05", "as the π0 timeline report"),
    ("E5_smolvla_timeline.md", "T008", "E5_timeline.md", "smolvla", "as the π0 timeline report"),
    ("E7_pi05_power.md", "T008", "E7_power_energy.md", "pi05", "as the π0 power report"),
    ("E7_smolvla_power.md", "T008", "E7_power_energy.md", "smolvla", "as the π0 power report"),
    ("E5_groot_timeline.md", "T009", "E5_timeline.md", "groot", "as T008 step 2"),
    ("E7_groot_power.md", "T009", "E7_power_energy.md", "groot", "as T008 step 5 (MAXN)"),
    ("E11_pi05_hybrid_inputs.md", "T011", "E11_models.md", "pi05", "the §13.8 hybrid-model inputs"),
    ("E11_smolvla_hybrid_inputs.md", "T011", "E11_models.md", "smolvla", "the §13.8 hybrid-model inputs"),
    ("E11_groot_n17_hybrid_inputs.md", "T012", "E11_models.md", "groot_n17", "the §13.8 hybrid-model inputs"),
]


def extract(src: str, key: str) -> tuple[list[str], int]:
    """Sections of `src` restricted to `key`. Returns (lines, number of kept data rows)."""
    rx = MATCH[key]
    lines = src.splitlines()
    out, kept = [], 0
    i, n = 0, len(lines)
    cur_h2 = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if cur_h2 and any(l.startswith("|") or l.startswith("###") for l in buf):
            out.append(cur_h2)
            out.append("")
            out.extend(buf)
            out.append("")
        buf = []

    skip_h3 = False
    while i < n:
        l = lines[i]
        if l.startswith("## "):
            flush()
            cur_h2, skip_h3 = l, False
            i += 1
            continue
        if cur_h2 is None and l.startswith("### "):
            # E5b_summary.md opens straight into per-kernel ### sections with no ## above
            # them; the first version skipped the whole file and found "no rows".
            cur_h2 = "## Per kernel"
        if cur_h2 is None:
            i += 1
            continue
        if l.startswith("### "):
            skip_h3 = not rx.search(l)
            if not skip_h3:
                buf.append(l)
            i += 1
            continue
        if l.startswith("|"):
            # a table: header, separator, rows
            tbl = []
            while i < n and lines[i].startswith("|"):
                tbl.append(lines[i])
                i += 1
            if skip_h3:
                continue
            if len(tbl) >= 2 and re.match(r"^\|[\s:|-]+\|$", tbl[1].replace("-", "-")):
                head, body = tbl[:2], tbl[2:]
            else:
                head, body = [], tbl
            # inside a model's own ### section every row belongs to it; otherwise filter
            in_model_h3 = bool(buf) and any(b.startswith("### ") for b in buf[-40:]) and not skip_h3 \
                and rx.search(next((b for b in reversed(buf) if b.startswith("### ")), ""))
            cells = [c.strip() for c in head[0].strip("|").split("|")] if head else []
            mcols = [j for j, c in enumerate(cells) if j > 0 and rx.search(c)]
            if mcols and not in_model_h3:
                # COLUMN-oriented table (E11_models.md: segments are rows, models columns).
                # Project onto the first column plus this model's columns and drop rows with
                # no value for it; a row filter would keep only rows that happen to NAME it.
                def proj(line):
                    cs = [c.strip() for c in line.strip().strip("|").split("|")]
                    return "| " + " | ".join(cs[j] for j in [0] + mcols) + " |"
                head = [proj(h) for h in head]
                rows = []
                for r in body:
                    cs = [c.strip() for c in r.strip().strip("|").split("|")]
                    if any(j < len(cs) and cs[j] not in ("—", "-", "") for j in mcols):
                        rows.append(proj(r))
            else:
                rows = body if in_model_h3 else [r for r in body if rx.search(r)]
            if rows:
                buf.extend(head + rows)
                buf.append("")
                kept += len(rows)
            continue
        i += 1
    flush()
    return out, kept


def main() -> int:
    made = []
    for fname, task, srcname, key, asks in TARGETS:
        dst = REPO / "reports" / fname
        if dst.exists() and "Generated by `bench/make_model_reports.py`" not in dst.read_text():
            print(f"  exists and was NOT generated here, not overwritten: {dst.name}")
            continue
        src = (REPO / "reports" / srcname).read_text()
        body, kept = extract(src, key)
        if kept == 0:
            print(f"  *** {fname}: no rows for {key} in {srcname} -- NOT written (an empty "
                  f"extract would satisfy the watchdog while delivering nothing)")
            continue
        head = [
            f"# {fname[:-3].replace('_', ' ')} — {LABEL[key]}",
            "",
            f"**Task {task}; acceptance names this file for {asks}. This is a VERBATIM EXTRACT "
            f"of `reports/{srcname}`**: every table keeps its header and only the rows for "
            f"{LABEL[key]}, and every per-configuration section kept is copied whole. Nothing "
            f"is recomputed, so it cannot disagree with its source. Cross-model prose is NOT "
            f"copied — it argues about several models at once — so read `reports/{srcname}` "
            f"for the commentary, conditions and caveats that apply to these numbers.",
            "",
            f"*Generated by `bench/make_model_reports.py`. {kept} data rows.*"
            + (f" The model's complete §13.8 hand-off, of which this is the summary view, is "
               f"`results/20260924_E11_{key}_inputs/model_inputs.csv`."
               if srcname == "E11_models.md" and key in ("pi05", "smolvla") else "")
            + (" GR00T's E11 is host-segment timing only (`results/20260924_E11a_groot_n17_"
               "hostseg/`); see E11_models.md's coverage section for which §13.8 quantities "
               "it has." if key == "groot_n17" else ""),
            "",
        ]
        dst.write_text("\n".join(head + body) + "\n")
        made.append((fname, kept))
        print(f"  wrote {fname:34s} {kept:4d} rows from {srcname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
