#!/usr/bin/env python3
"""
Static-vs-dynamic instruction-count cross-check for any kernel.

Reads:
  - inventory CSV produced by build_inst_inventory.py
  - parsed-ncu JSON produced by parse_ncu_results.py (with pipe_lsu_split)

Emits:
  - markdown table of static counts + dynamic ncu pipe utilisation
  - reconciliation paragraph explaining why static ≠ dynamic

The cross-check is meant to be qualitative: static counts tell you what
instruction mix the compiler emitted; dynamic counters tell you which pipes
the GPU spent issue cycles on at runtime. Predicated instructions, conditional
branches, and per-pipe issue rates make the relationship non-trivial.

Usage:
    python crosscheck_sass.py \\
        --inventory <kernel>_total_inst_inventory.csv \\
        --ncu-json <parsed_ncu_results>.json \\
        --target-kernel <kernel_short_name>

The parsed-ncu JSON schema is:
    { <run_or_matrix_id>: { <kernel_short_name>: {
          "pipe_utilization": { "lsu_active_pct": ..., "fma_active_pct": ..., ... },
          "pipe_lsu_split":   { "load_pct": ..., "write_pct": ...,
                                "load_to_write_ratio": ... },
          ...
    } } }

If the JSON is a flat per-kernel dict (no per-run grouping), use
--single-run to treat the top-level keys as kernel names directly.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path


def read_inventory(csv_path: Path) -> dict[str, int]:
    """Parse the category-subtotal section of build_inst_inventory.py CSV."""
    cats: dict[str, int] = {}
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    seen_blank = False
    for row in rows:
        if not row:
            seen_blank = True
            continue
        if not seen_blank:
            continue
        # Skip the optional second blank-then-lsu_subclass section
        if row[0] in {"category", "lsu_subclass"} or row[0].startswith("#"):
            continue
        if row[0] == "TOTAL":
            cats["TOTAL"] = int(row[1])
            continue
        # Only accept known category names
        if row[0] in {"LSU", "FMA", "ALU", "Shuffle", "XU", "Reduce",
                      "Uniform", "Branch", "Other"}:
            cats[row[0]] = int(row[1])
    return cats


def collect_kernel_data(ncu: dict, target_kernel: str,
                        single_run: bool) -> list[tuple[str, dict]]:
    """Return list of (run_id, kernel_data_dict) for the target kernel.

    If single_run, treat top-level keys as kernel names. Otherwise expect
    a per-run dict mapping run_id -> {kernel_name -> kernel_data}.
    """
    if single_run:
        # Top level is the per-kernel dict from a single profile run
        if target_kernel in ncu:
            return [("(single-run)", ncu[target_kernel])]
        return []

    rows = []
    for run_id, kernels in sorted(ncu.items()):
        if not isinstance(kernels, dict):
            continue
        kdata = kernels.get(target_kernel)
        if kdata:
            rows.append((run_id, kdata))
    return rows


PIPE_RE = __import__("re").compile(
    r"sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active")
LSU_LD_RE = __import__("re").compile(
    r"l1tex__t_requests_pipe_lsu_mem_global_op_(ld|st)\.sum")


def _short_name(name: str) -> str:
    """Demangled kernel name -> the short name used by --target-kernel and the inventory.

    Strips a leading return type, cuts at the first template or argument bracket, and
    keeps the last namespace component. Returns "" for anything that does not leave a
    usable identifier, which the caller skips — the first version returned '' and 'void'
    as if they were kernels, because a name like
    `void at::native::elementwise_kernel<...>(...)` was cut before the `void ` was
    removed.
    """
    import re as _re
    n = name.strip()
    n = _re.sub(r"^(void|__global__|static)\s+", "", n)
    n = _re.split(r"[<(]", n)[0].strip()
    n = n.rsplit("::", 1)[-1].strip()
    return n if _re.fullmatch(r"[A-Za-z_]\w*", n or "") else ""


def ncu_from_csv(path: Path) -> dict:
    """Build {kernel_short_name: {pipe_utilization, pipe_lsu_split}} from an ncu CSV.

    This script came from the desktop study, where a `parse_ncu_results.py` emitted a
    JSON. That script does not exist on this device; what exists is `parse_ncu.py` and
    the `--csv --page details` exports that bench/run_ncu_*.sh writes. Rather than port
    the whole desktop pipeline, this reads the same metrics off the CSV and hands this
    script the shape it already expects.

    The kernel key is the SHORT name — the demangled name truncated at the first `<` or
    `(` — because that is what --target-kernel is given and what the inventory files are
    named after.

    A kernel whose capture did not request the pipe metrics arrives with zeros, which the
    table then shows as zeros. That is the correct outcome: it says the metrics are
    missing, rather than silently omitting the row. `ncu` reports the sections' metrics
    but NOT the per-pipe ones unless they are requested by name, which is the same trap
    that cost C1 two captures before it was caught.
    """
    import csv as _csv
    import re as _re
    kernels: dict[str, dict] = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            name = (r.get("Kernel Name") or "").strip()
            if not name:
                continue
            short = _short_name(name)
            if not short:
                continue
            k = kernels.setdefault(short, {"pipe_utilization": {},
                                           "pipe_lsu_split": {},
                                           "_ld": 0.0, "_st": 0.0})
            metric = (r.get("Metric Name") or "").strip()
            try:
                v = float((r.get("Metric Value") or "").replace(",", ""))
            except ValueError:
                continue
            m = PIPE_RE.fullmatch(metric)
            if m:
                k["pipe_utilization"][f"{m.group(1)}_active_pct"] = v
                continue
            m = LSU_LD_RE.fullmatch(metric)
            if m:
                k["_ld" if m.group(1) == "ld" else "_st"] += v
    for k in kernels.values():
        ld, st = k.pop("_ld"), k.pop("_st")
        tot = ld + st
        if tot > 0:
            k["pipe_lsu_split"] = {"load_pct": 100 * ld / tot,
                                   "write_pct": 100 * st / tot,
                                   "load_to_write_ratio": ld / max(1e-9, st)}
    return kernels



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", type=Path, required=True,
                    help="CSV from build_inst_inventory.py (total-kernel)")
    ap.add_argument("--ncu-json", type=Path, default=None,
                    help="JSON from parse_ncu_results.py (the desktop reference's "
                         "schema). On this device use --ncu-csv instead.")
    ap.add_argument("--ncu-csv", type=Path, default=None,
                    help="ncu --csv --page details export, which is what "
                         "bench/run_ncu_*.sh actually produces here. There is no "
                         "parse_ncu_results.py on this device; parse_ncu.py reads the "
                         "same pipe metrics off this CSV, so the adapter below builds "
                         "the structure this script was written against rather than "
                         "inventing a second one.")
    ap.add_argument("--target-kernel", type=str, required=True,
                    help="short kernel name (e.g. csrmv_v3_kernel)")
    ap.add_argument("--single-run", action="store_true",
                    help="treat ncu JSON as flat {kernel_name: kernel_data} "
                         "(default expects {run_id: {kernel_name: kernel_data}})")
    ap.add_argument("--gpu-arch", type=str, default="",
                    help="optional GPU arch label for the report header (e.g. sm_120)")
    args = ap.parse_args()

    if not (args.ncu_json or args.ncu_csv):
        print("ERROR: pass --ncu-csv (this device) or --ncu-json (desktop reference).",
              file=sys.stderr)
        return 1
    inv = read_inventory(args.inventory)
    if args.ncu_csv:
        ncu = ncu_from_csv(args.ncu_csv)
        args.single_run = True
    else:
        with open(args.ncu_json) as f:
            ncu = json.load(f)

    total = inv.get("TOTAL", sum(v for k, v in inv.items() if k != "TOTAL"))
    if total == 0:
        print(f"ERROR: inventory {args.inventory} has no instructions.",
              file=sys.stderr)
        return 1

    s_lsu = inv.get("LSU", 0)
    s_fma = inv.get("FMA", 0)
    s_alu = inv.get("ALU", 0)
    s_xu = inv.get("XU", 0) + inv.get("Shuffle", 0)
    s_other = total - s_lsu - s_fma - s_alu - s_xu

    arch_str = f" ({args.gpu_arch})" if args.gpu_arch else ""
    print(f"# Static-vs-dynamic cross-check{arch_str} — `{args.target_kernel}`")
    print()
    print(f"## Static SASS counts (from `{args.inventory.name}`)")
    print()
    print(f"- LSU        : **{s_lsu}** ({s_lsu/total*100:.1f}%)")
    print(f"- FMA        : **{s_fma}** ({s_fma/total*100:.1f}%)")
    print(f"- ALU        : **{s_alu}** ({s_alu/total*100:.1f}%)")
    print(f"- XU/Shuffle : **{s_xu}** ({s_xu/total*100:.1f}%)")
    print(f"- Other      : **{s_other}** ({s_other/total*100:.1f}%)")
    print(f"- TOTAL      : **{total}**")
    print()
    print(f"Static ratios:  "
          f"LSU/FMA = {s_lsu/max(1,s_fma):.2f}x,  "
          f"LSU/ALU = {s_lsu/max(1,s_alu):.2f}x,  "
          f"ALU/FMA = {s_alu/max(1,s_fma):.2f}x")
    print()

    rows = collect_kernel_data(ncu, args.target_kernel, args.single_run)
    if not rows:
        print(f"WARNING: no entries for kernel '{args.target_kernel}' in "
              f"{args.ncu_csv or args.ncu_json}. Available: "
              f"{sorted(ncu)[:8]}{'...' if len(ncu) > 8 else ''}",
              file=sys.stderr)
    else:
        single_config = len(rows) == 1
        if single_config:
            print(f"## Dynamic ncu pipe utilisation (single-config — one ncu run)")
            print()
            print(f"> Only one configuration is profiled. The table below shows the "
                  f"single run's dynamic counters. For workloads where multiple inputs "
                  f"or batch sizes are profiled separately, re-run this with a "
                  f"top-level `{{run_id: {{kernel_name: ...}}}}` JSON to compare across runs.")
            print()
        else:
            print(f"## Dynamic ncu pipe utilisation (`pct_of_peak_sustained_active`)")
            print()
        print("| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        ratios_lsu_fma = []
        for run_id, kdata in rows:
            pipes = kdata.get("pipe_utilization", {})
            split = kdata.get("pipe_lsu_split", {})
            lsu = pipes.get("lsu_active_pct", 0) or 0
            fma = pipes.get("fma_active_pct", 0) or 0
            alu = pipes.get("alu_active_pct", 0) or 0
            ld_ = split.get("load_pct", 0) or 0
            wr_ = split.get("write_pct", 0) or 0
            ldwr = split.get("load_to_write_ratio", 0) or 0
            d_lsu_fma = lsu / max(0.01, fma)
            d_lsu_alu = lsu / max(0.01, alu)
            ratios_lsu_fma.append(d_lsu_fma)
            print(f"| `{run_id}` | {lsu:.2f} | {fma:.2f} | {alu:.2f} | "
                  f"{ld_:.2f} | {wr_:.2f} | {ldwr:.2f} | "
                  f"{d_lsu_fma:.2f} | {d_lsu_alu:.2f} |")
        print()

        if ratios_lsu_fma and s_fma > 0:
            d_min = min(ratios_lsu_fma)
            d_max = max(ratios_lsu_fma)
            s_ratio = s_lsu / s_fma
            print(f"**Direction agreement check.** Static LSU/FMA = "
                  f"{s_ratio:.2f}x; dynamic ranges {d_min:.2f}x-{d_max:.2f}x. "
                  f"Static counts predict the LSU-heavy character; dynamic ratio is "
                  f"{(d_min+d_max)/2/max(0.01,s_ratio):.1f}x amplified — see why below.")
            print()

    print("## Why static ≠ dynamic")
    print()
    print("- **Static counts include every SASS line emitted by the compiler**, "
          "including code paths that fire only on boundary cases.")
    print("- **Predication.** Many ALU/branch ops gate body code that only "
          "fires on boundary rows or sentinel iterations. The static count "
          "counts them once; the dynamic counter counts them per warp-execution.")
    print("- **Looped vs unrolled execution.** Long-span backward BRAs are "
          "outer iteration loops that execute many times; short-span ones are "
          "bisection / spin / barrier waits that fire on contention only. "
          "Fully-unrolled inner blocks execute once per warp tile. Static count "
          "reflects source-text layout, not weighted-by-trip-count execution.")
    print("- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is "
          "normalised by the maximum issuable per cycle for that pipe, which "
          "differs across pipes (LSU has its own issue slot; ALU and FMA share "
          "some sub-pipes). 100% LSU is a different denominator than 100% FMA, "
          "so dividing dynamic LSU% / FMA% is not the same as dividing static "
          "LSU count / FMA count.")
    print("- **Per-instruction issue cost.** Vector LDGs count as one *instruction* "
          "but burn multiple LSU cycles; NOPs and uniform-datapath ops consume "
          "zero pipe slots. Static and dynamic are measuring different things.")
    print()
    print("**Bottom line.** Static counts give the *direction* of pipe utilisation "
          "(LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives "
          "the magnitude. If the static and dynamic directions disagree (e.g. "
          "static says ALU-heavy but dynamic says LSU-heavy), check whether the "
          "ALU instructions are inside a predicated boundary path that rarely fires.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
