"""T004 / T007 — turn ncu's exports into the tables the protocol asks for.

Spec: `docs/03_instruction_level_analysis.md` §11 item 2. Produces, per region:

  `ncu/columns.txt`             the exported header and the distinct Metric Names (§3.3)
  `ncu/kernels.csv`             one row per kernel with the metrics we are allowed to quote
  `ncu/region_utilisation.csv`  region, n_kernels_captured, kernel_time_ms,
                                kernel_time_coverage, metric, kernel_weighted, region_weighted
  `ncu/parsed.json`             the per-kernel dict E5b consumes

Three [Orin] rules from protocol §3.3 are enforced here rather than left to the writer:

  * **There are no DRAM counters.** `dram__*` does not exist on this device; nothing derived
    from one is emitted.
  * **`Memory Throughput` is L2 throughput**, not off-chip bandwidth. It is renamed
    `memory_l2_throughput_pct` so a table cannot accidentally read it as DRAM.
  * **Off-chip bytes come from the L2 sysmem-miss sectors** —
    `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum * 32 B` for reads and, as T005
    established, `lts__t_sectors_aperture_sysmem_op_write.sum * 32 B` for writes, because the
    write-miss counter is identically zero on this device.

`region_weighted` uses **nsys** durations and region time with idle gaps counted as zero, and
is labelled derived — it is the number to compare with the paper's "SM utilisation 22 %".

    python bench/parse_ncu.py <run-dir> --region "ae_layer[0]" [--nsys-region-ms 12.3]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json  # noqa: E402

SECTOR = 32
STALL_RE = re.compile(
    r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
PIPE_RE = re.compile(r"sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active")
KEEP = {
    "Duration": "duration",
    "Compute (SM) Throughput": "compute_sm_throughput_pct",
    "Memory Throughput": "memory_l2_throughput_pct",   # [Orin] L2, never DRAM
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Achieved Active Warps Per SM": "achieved_active_warps_per_sm",
    "Executed Ipc Active": "executed_ipc_active",
    "L2 Hit Rate": "l2_hit_rate_pct",
    "SM Frequency": "sm_frequency",
    "Block Size": "block_size",
    "Grid Size": "grid_size",
}
SYSMEM = {
    "lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum": "sysmem_read_miss_sectors",
    "lts__t_sectors_aperture_sysmem_op_write.sum": "sysmem_write_sectors",
    "lts__t_sectors_aperture_sysmem_lookup_miss.sum": "sysmem_miss_sectors_all",
}


def num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def read_page(path: Path) -> list[dict]:
    """ncu --csv writes a `==PROF==` preamble; both the long (details) and wide (raw) shapes
    are handled. Parsed with csv.DictReader because kernel names contain commas (§3.2)."""
    lines = [ln for ln in path.read_text(errors="replace").splitlines()
             if not ln.startswith("==")]
    return list(csv.DictReader(lines)) if lines else []


def collect(rows: list[dict]) -> tuple[dict, set]:
    """-> {kernel_id: {metric: value}}, and the set of distinct metric names seen."""
    out: dict[str, dict] = {}
    names: set[str] = set()
    if not rows:
        return out, names
    long_form = "Metric Name" in rows[0]
    for r in rows:
        kid = r.get("ID") or r.get("Kernel Name", "")
        k = out.setdefault(kid, {"kernel_name": r.get("Kernel Name", ""),
                                 "block_size": r.get("Block Size", ""),
                                 "grid_size": r.get("Grid Size", "")})
        if long_form:
            m = r.get("Metric Name", "")
            names.add(m)
            v = num(r.get("Metric Value"))
            if v is None:
                continue
            k[m] = v
            k.setdefault("_units", {})[m] = r.get("Metric Unit", "")
        else:
            for m, v in r.items():
                if m in ("ID", "Kernel Name", "Process ID", "Process Name", "Host Name",
                         "Context", "Stream", "Device", "CC"):
                    continue
                names.add(m)
                vv = num(v)
                if vv is not None:
                    k[m] = vv
    return out, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--region", required=True)
    ap.add_argument("--nsys-region-ms", type=float, default=None,
                    help="the region's wall time from nsys, for region_weighted (§3.3)")
    ap.add_argument("--prefix", default=None,
                    help="basename of the ncu exports; defaults to a sanitised region name")
    args = ap.parse_args()
    d = Path(args.rundir)
    ncu = d / "ncu"
    tag = args.prefix or re.sub(r"[^A-Za-z0-9]+", "_", args.region).strip("_")

    merged, names = {}, set()
    for page in ("details", "raw"):
        f = next(iter(sorted(ncu.glob(f"{tag}*{page}*.csv"))), None)
        if f is None:
            print(f"  {page} page missing for {tag}")
            continue
        rows = read_page(f)
        part, nm = collect(rows)
        names |= nm
        for kid, vals in part.items():
            merged.setdefault(kid, {}).update(vals)
        print(f"  {page}: {len(part)} kernels, {len(nm)} metric names from {f.name}")
    if not merged:
        print("no ncu data found")
        return 1

    (ncu / "columns.txt").write_text(
        f"# region: {args.region}\n# distinct Metric Name values in the exports\n"
        + "\n".join(sorted(names)) + "\n")

    krows, parsed = [], {}
    for kid, v in merged.items():
        row = {"region": args.region, "id": kid, "kernel_name": v.get("kernel_name", ""),
               "block_size": v.get("block_size", ""), "grid_size": v.get("grid_size", "")}
        for src, dst in KEEP.items():
            if src in v:
                row[dst] = v[src]
        for src, dst in SYSMEM.items():
            if src in v:
                row[dst] = v[src]
        rb = v.get("lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum")
        wb = v.get("lts__t_sectors_aperture_sysmem_op_write.sum")
        if rb is not None:
            row["offchip_read_bytes"] = int(rb * SECTOR)
        if wb is not None:
            row["offchip_write_bytes"] = int(wb * SECTOR)
        stalls = {}
        for m, val in v.items():
            s = STALL_RE.match(str(m))
            if s:
                stalls[s.group(1)] = val
            p = PIPE_RE.match(str(m))
            if p:
                row[f"pipe_{p.group(1)}_pct"] = val
        if stalls:
            tot = sum(stalls.values())
            row["stall_sum"] = round(tot, 4)
            row["stall_reasons_n"] = len(stalls)
            for k_, val in stalls.items():
                row[f"stall_{k_}"] = val
        krows.append(row)
        parsed[kid] = {"kernel_name": row["kernel_name"], "metrics": row, "stalls": stalls}

    keys = sorted({k for r in krows for k in r})
    with (ncu / "kernels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in krows])

    # --- region_utilisation.csv
    dur = [r.get("duration") for r in krows if r.get("duration") is not None]
    kernel_ms = sum(dur) / 1e6 if dur else 0.0            # ncu Duration is ns on this build
    cov = (kernel_ms / args.nsys_region_ms) if args.nsys_region_ms else ""
    urows = []
    metrics = ["compute_sm_throughput_pct", "memory_l2_throughput_pct",
               "achieved_occupancy_pct", "achieved_active_warps_per_sm",
               "executed_ipc_active", "l2_hit_rate_pct"]
    metrics += sorted({k for r in krows for k in r if k.startswith(("pipe_", "stall_"))
                       and k not in ("stall_sum", "stall_reasons_n")})
    for m in metrics:
        vals = [(r.get(m), r.get("duration")) for r in krows
                if r.get(m) is not None and r.get("duration")]
        if not vals:
            continue
        tw = sum(d for _, d in vals)
        kw = sum(v * d for v, d in vals) / tw if tw else ""
        rw = (kw * kernel_ms / args.nsys_region_ms) if args.nsys_region_ms else ""
        urows.append({"region": args.region, "n_kernels_captured": len(krows),
                      "kernel_time_ms": round(kernel_ms, 6),
                      "kernel_time_coverage": round(cov, 4) if cov != "" else "",
                      "metric": m,
                      "kernel_weighted": round(kw, 6) if kw != "" else "",
                      "region_weighted": round(rw, 6) if rw != "" else ""})
    with (ncu / "region_utilisation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "n_kernels_captured", "kernel_time_ms",
                                          "kernel_time_coverage", "metric",
                                          "kernel_weighted", "region_weighted"])
        w.writeheader()
        w.writerows(urows)

    sums = [r["stall_sum"] for r in krows if "stall_sum" in r]
    write_json(ncu / "parsed.json", {
        "region": args.region, "n_kernels": len(krows),
        "kernel_time_ms": kernel_ms, "kernel_time_coverage": cov,
        "nsys_region_ms": args.nsys_region_ms,
        "stall_sum_median": sorted(sums)[len(sums) // 2] if sums else None,
        "stall_reasons_n": krows[0].get("stall_reasons_n") if krows else None,
        "notes": [
            "[Orin] Memory Throughput is L2, not DRAM; emitted as memory_l2_throughput_pct.",
            "[Orin] no dram__* counters exist; off-chip bytes are sysmem-miss sectors x 32 B "
            "for reads and op_write.sum x 32 B for writes (T005 found the write-miss counter "
            "is identically zero).",
            "region_weighted uses the nsys region time with idle gaps counted as zero and is "
            "DERIVED; it is the figure comparable with the paper's 22 % SM utilisation.",
            "Stall ratios are reported as exported; their sum per kernel is recorded rather "
            "than renormalised (§3.3).",
        ],
        "kernels": parsed})
    print(f"  {len(krows)} kernels, kernel time {kernel_ms:.4f} ms"
          + (f", coverage {cov:.1%}" if cov != "" else "")
          + (f", stall sums median {sorted(sums)[len(sums)//2]:.3f}" if sums else ""))
    print(f"wrote {ncu}/kernels.csv, region_utilisation.csv, parsed.json, columns.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
