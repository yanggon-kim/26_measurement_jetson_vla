"""T001 step 4d: ncu metric discovery on this device (docs/02_protocol.md §3.3).

list     -> writes the explicit metric list (one ncu name per line) for form D: every metric protocol §3.3 names,
            plus the candidates of `ncu --query-metrics --chip <chip>` for the step-4d patterns.
analyze  -> joins the list with the raw-page CSV of the form-D toy capture into metric_presence.csv
            (metric, category, present, nonzero_on_toy, notes).

python bench/metric_discovery.py list    --query env/tool_help/ncu_query_metrics_chip_gb10b.txt --out <list.txt>
python bench/metric_discovery.py analyze --query ... --list <list.txt> --raw <toy_metrics_raw.csv> --out <presence.csv>
"""
import argparse
import csv
import re

# protocol §3.3 names (base names; suffixes are added from the metric type unless given)
PROTOCOL = {
    "dram_bytes": ["dram__bytes_read.sum", "dram__bytes_write.sum", "dram__sectors_read.sum", "dram__sectors_write.sum"],
    "l2_sysmem_bytes": ["lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum",
                        "lts__t_sectors_aperture_sysmem_op_write.sum", "lts__d_sectors_fill_sysmem.sum",
                        "lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss.sum"],
    "simt_flops": [f"smsp__sass_thread_inst_executed_op_{p}{o}_pred_on.sum"
                   for p in ("f", "h", "d") for o in ("add", "mul", "fma")],
    "lsu_requests": [f"l1tex__t_requests_pipe_lsu_mem_global_op_{o}.sum" for o in ("ld", "st", "red", "atom")]
                    + [f"l1tex__t_requests_pipe_lsu_mem_local_op_{o}.sum" for o in ("ld", "st")]
                    + [f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{o}.sum" for o in ("ld", "st", "atom")],
    "e5b_launch": ["gpc__cycles_active.avg", "gpc__cycles_active.avg.per_warp", "sm__warps_launched.sum",
                   "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
                   "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
                   "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps"],
    "occupancy_activity": ["sm__cycles_active.avg", "sm__warps_active.avg.pct_of_peak_sustained_active",
                           "sm__maximum_warps_per_active_cycle_pct", "sm__warps_active.avg.per_cycle_active",
                           "sm__instruction_throughput.avg.pct_of_peak_sustained_active"],
    "throughput": ["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                   "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                   "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
                   "sm__inst_executed.avg.per_cycle_active", "gpc__cycles_elapsed.max",
                   "gpc__cycles_elapsed.avg.per_second", "sm__cycles_elapsed.avg.per_second"],
}
# step-4d candidate patterns (matched against the chip's metric list)
PATTERNS = [
    ("dram_bytes", r"^dram__"),
    ("l2_sysmem_bytes", r"^lts__t_sectors_(srcunit_tex_)?aperture_sysmem(_op_(read|write))?(_lookup_(hit|miss))?$"),
    ("l2_sysmem_bytes", r"^lts__d_sectors_fill"),
    ("memory_controller", r"^mcc__"),
    ("tensor_pipe", r"^sm__pipe_tensor"),
    ("tensor_pipe", r"^sm__inst_executed_pipe_(tensor|tc|tmem|tma)"),
    ("tensor_flops", r"^sm__ops_path_tensor"),
    ("pipe_mix", r"^sm__inst_executed_pipe_(lsu|fma|alu|xu|adu|cbu|fmaheavy|fmalite|fma_type_fp16|fp64|uniform|tex)$"),
    ("pipe_mix", r"^sm__pipe_(fma|alu|lsu|xu|fp16|shared|fmaheavy|fp64)_cycles_active$"),
    ("simt_flops", r"^smsp__sass_thread_inst_executed_op_"),
    ("warp_stalls", r"^smsp__average_warps_issue_stalled_"),
    ("lsu_requests", r"^l1tex__t_requests_pipe_lsu"),
    ("e5b_launch", r"^gpc__cycles_active"),
]
SUFFIX = {"Counter": ".sum", "Ratio": ".ratio", "Throughput": ".avg.pct_of_peak_sustained_elapsed"}


def parse_query(path):
    out = {}
    for line in open(path):
        tok = line.split()
        if len(tok) >= 2 and re.match(r"^[a-z0-9]+__[a-z0-9_]+$", tok[0]) and tok[1] in SUFFIX:
            out[tok[0]] = tok[1]
    return out


def build_list(query):
    rows = {}  # full name -> (category, note)
    for cat, names in PROTOCOL.items():
        for n in names:
            rows.setdefault(n, (cat, "protocol §3.3"))
    for cat, pat in PATTERNS:
        for base, typ in query.items():
            if re.search(pat, base):
                rows.setdefault(base + SUFFIX[typ], (cat, f"query candidate ({typ})"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "analyze"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--list")
    ap.add_argument("--raw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    query = parse_query(a.query)
    rows = build_list(query)
    if a.cmd == "list":
        # only names whose base exists for this chip go to ncu (unknown names abort the whole collection);
        # launch__* and derived suffixes are kept because they are not in the chip list
        keep = [n for n in rows if n.split(".")[0] in query or n.startswith("launch__")]
        open(a.out, "w").write("\n".join(keep) + "\n")
        print(f"{len(keep)} metrics to collect; {len(rows) - len(keep)} named but not in the chip list")
        return
    asked = set(open(a.list).read().split())
    raw = list(csv.reader(open(a.raw)))
    hdr, val = raw[0], raw[2]
    got = dict(zip(hdr, val))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "category", "present", "nonzero_on_toy", "notes"])
        for n, (cat, note) in rows.items():
            base = n.split(".")[0]
            in_chip = base in query or n.startswith("launch__")
            if n in got:
                v = got[n].replace(",", "")
                try:
                    nz = float(v) != 0.0
                except ValueError:
                    nz = v not in ("", "n/a")
                w.writerow([n, cat, "yes", "yes" if nz else "no", f"{note}; toy value {got[n]}"])
            elif not in_chip:
                w.writerow([n, cat, "no", "n/a", f"{note}; not in `ncu --query-metrics --chip` list"])
            elif n in asked:
                w.writerow([n, cat, "no", "n/a", f"{note}; requested but no column returned"])
            else:
                w.writerow([n, cat, "unknown", "n/a", f"{note}; not requested"])


if __name__ == "__main__":
    main()
