"""T001 step 3c — what an ncu export actually contains on this device.

Reads the two exported pages of one .ncu-rep CSV export and writes
  <ncu_dir>/columns.txt            exact headers + the distinct metric names
  <ncu_dir>/metric_presence.csv    every protocol §3.3 metric: present / absent / empty

Parsing uses csv.DictReader throughout (protocol §3.2: kernel names contain commas).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# The metrics protocol §3.3 asks us to keep. Names as written there; the point of
# this script is to report which of them this Orin + this ncu version actually export.
DETAILS_WANTED = {
    "occupancy": ["SM Busy", "SM Active Cycles", "Achieved Occupancy",
                  "Theoretical Occupancy", "Achieved Active Warps Per SM"],
    "throughput": ["Duration", "Compute (SM) Throughput", "Memory Throughput",
                   "DRAM Throughput", "DRAM Utilization", "Mem Busy", "L2 Hit Rate",
                   "L1/TEX Hit Rate", "Executed Ipc Active", "Executed Ipc Elapsed"],
}
RAW_WANTED = {
    "occupancy_raw": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "pipe_active": [f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_active"
                    for p in ("lsu", "fma", "alu", "xu", "adu", "cbu", "fp16")],
    "pipe_elapsed": [f"sm__inst_executed_pipe_{p}.avg.pct_of_peak_sustained_elapsed"
                     for p in ("lsu", "fma", "alu", "xu")],
    "lsu_requests": [
        f"l1tex__t_requests_pipe_lsu_mem_global_op_{o}.sum"
        for o in ("ld", "st", "red", "atom")
    ] + [
        f"l1tex__t_requests_pipe_lsu_mem_local_op_{o}.sum" for o in ("ld", "st")
    ] + [
        f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{o}.sum"
        for o in ("ld", "st", "atom")
    ],
    "e5b": ["gpc__cycles_active.avg.per_warp", "launch__grid_size", "launch__block_size",
            "launch__registers_per_thread", "launch__occupancy_limit_blocks",
            "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
            "launch__occupancy_limit_warps"],
    # the 18 warp-stall reasons as this ncu version names them on the raw page.
    # The details page has a "Warp State Statistics" section but exports no
    # "Stall ..." display metric, so these raw names are what E5/E5b must use.
    "stall_reasons_raw": [
        f"smsp__average_warps_issue_stalled_{r}_per_issue_active.ratio" for r in (
            "barrier", "branch_resolving", "dispatch_stall", "drain", "imc_miss",
            "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
            "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
            "short_scoreboard", "sleeping", "tex_throttle", "wait")
    ],
    # DRAM counters: expected to be dead on Tegra (unified LPDDR5 behind the L2).
    # If they are, off-chip traffic must come from the L2 sysmem-miss counters below.
    "dram": ["dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
             "dram__cycles_active.sum", "dram__cycles_elapsed.sum",
             "dram__bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"],
    "dram_proxy_l2": ["lts__t_sectors_aperture_sysmem_lookup_miss.sum",
                      "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum",
                      "lts__t_sectors_aperture_sysmem_op_write.sum",
                      "lts__t_sectors_lookup_miss.sum", "lts__t_sectors.sum",
                      "lts__t_sector_hit_rate.pct",
                      "lts__throughput.avg.pct_of_peak_sustained_elapsed"],
}

# Metrics the protocol names that this device does not export, and what to use instead.
SUBSTITUTES = {
    "DRAM Throughput": "no substitute in ncu: every dram__* counter reads 0 on this Tegra. "
                       "Use lts__t_sectors_aperture_sysmem_*.sum x 32 B as measured off-chip traffic.",
    "DRAM Utilization": "as above; ncu's 'Memory Throughput' on this device equals "
                        "lts__throughput (L2), not DRAM.",
    "sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active":
        "not exported; bf16/fp16 MMA work shows in sm__pipe_tensor_op_hmma_cycles_active.* instead.",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed":
        "use the .pct_of_peak_sustained_active variant, which is exported.",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed":
        "use the .pct_of_peak_sustained_active variant, which is exported.",
    "gpc__cycles_active.avg.per_warp":
        "not exported; derive from gpc__cycles_active.avg / sm__warps_launched.sum if needed.",
}


def read_details(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def read_raw(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """The raw page has a header row, a units row, then one row per kernel launch."""
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], [], []
    header, units = rows[0], (rows[1] if len(rows) > 1 else [])
    data = [dict(zip(header, r)) for r in rows[2:]]
    return header, units, data


def base_name(col: str) -> str:
    """Raw columns are prefixed, e.g. 'SM_A.TriageAC.sm__cycles_active.avg'."""
    return col.split(".", 2)[-1] if col.count(".") >= 2 and col[:2].isupper() else col


def classify(values: list[str]) -> str:
    """The three verdicts T001 step 3c asks for, plus the one the data forced.

    absent       the metric is not a column / Metric Name of the export at all
    empty        the column exists but every value is blank or 'n/a'
    present_zero the column exists and every value is exactly 0 — the counter may be
                 genuinely idle for this kernel (no local memory, no atomics) or may be
                 dead on this device (every dram__* is in this class). A zero on one
                 kernel is NOT evidence that a counter works; check per group.
    present      the column exists and carries a non-zero value
    """
    if not values:
        return "absent"
    non_empty = [v for v in values if v.strip() not in ("", "n/a", "N/A")]
    if not non_empty:
        return "empty"
    def is_zero(v: str) -> bool:
        try:
            return float(v.replace(",", "")) == 0.0
        except ValueError:
            return False
    return "present_zero" if all(is_zero(v) for v in non_empty) else "present"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="toy")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    d_hdr, d_rows = read_details(Path(args.details))
    r_hdr, r_units, r_rows = read_raw(Path(args.raw))

    metric_names = sorted({r["Metric Name"] for r in d_rows if r.get("Metric Name")})
    sections = sorted({r["Section Name"] for r in d_rows if r.get("Section Name")})
    # value per metric name (first kernel launch) so we can tell empty from absent
    d_values: dict[str, list[str]] = {}
    for r in d_rows:
        d_values.setdefault(r.get("Metric Name", ""), []).append(r.get("Metric Value", ""))
    r_values = {c: [row.get(c, "") for row in r_rows] for c in r_hdr}
    r_by_base: dict[str, list[str]] = {}
    for c in r_hdr:
        r_by_base.setdefault(base_name(c), []).extend(r_values[c])

    cols = out / "columns.txt"
    with cols.open("w") as f:
        f.write(f"# ncu CSV export column and metric inventory — {args.label}\n")
        f.write(f"# details: {args.details}\n# raw:     {args.raw}\n")
        f.write(f"# kernel launches in export: details={len({r['ID'] for r in d_rows})} "
                f"raw={len(r_rows)}\n\n")
        f.write(f"## details page header ({len(d_hdr)} columns)\n")
        for c in d_hdr:
            f.write(f"{c}\n")
        f.write(f"\n## details page: Section Name values ({len(sections)})\n")
        for s in sections:
            f.write(f"{s}\n")
        f.write(f"\n## details page: distinct Metric Name values ({len(metric_names)})\n")
        for m in metric_names:
            f.write(f"{m}\n")
        f.write(f"\n## raw page header ({len(r_hdr)} columns; units row below each)\n")
        for c, u in zip(r_hdr, r_units + [""] * len(r_hdr)):
            f.write(f"{c}\t[{u}]\n")
    print(f"wrote {cols}")

    pres = out / "metric_presence.csv"
    with pres.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page", "group", "metric_protocol_name", "status",
                    "exported_column_or_name", "example_value",
                    "substitute_on_this_device"])
        for group, names in DETAILS_WANTED.items():
            for n in names:
                st = classify(d_values.get(n, []))
                ex = next((v for v in d_values.get(n, []) if v.strip()), "")
                w.writerow(["details", group, n, st, n if st != "absent" else "", ex,
                            SUBSTITUTES.get(n, "")])
        # every 'Stall ...' metric the device exported
        for n in metric_names:
            if n.lower().startswith("stall") or "stall" in n.lower():
                ex = next((v for v in d_values.get(n, []) if v.strip()), "")
                w.writerow(["details", "stall_reasons_display", n,
                            classify(d_values[n]), n, ex, ""])
        for group, names in RAW_WANTED.items():
            for n in names:
                # a metric can appear under several prefixed column spellings
                # (e.g. "LTS.TriageSCG.lts__throughput..." next to the plain name);
                # take the one that actually carries a value.
                matches = [c for c in r_hdr if base_name(c) == n or c == n]
                if not matches:
                    w.writerow(["raw", group, n, "absent", "", "", SUBSTITUTES.get(n, "")])
                    continue
                best = next((c for c in matches
                             if any(v.strip() for v in r_values[c])), matches[0])
                vals = r_values[best]
                ex = next((v for v in vals if v.strip()), "")
                w.writerow(["raw", group, n, classify(vals), best, ex,
                            SUBSTITUTES.get(n, "")])
    print(f"wrote {pres}")

    # short console summary
    import collections
    counts = collections.Counter()
    with pres.open() as f:
        for row in csv.DictReader(f):
            counts[row["status"]] += 1
    print("status counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
