"""Per-variant DRAM bytes per frame from bench/run_ncu_e14.sh's raw ncu CSV (E12b columns + variant).
reads = L2 fills from memory (device + sysmem), writes <= L2 write sectors; 32 B per sector.
<variant>_expert ranges drop their first two graph workloads (the harness's KV-buffer prefix fills)."""
import csv
import json
import sys

UNIT = {"sector": 32, "byte": 1, "usecond": 1e-3, "nsecond": 1e-6, "msecond": 1.0, "us": 1e-3, "ms": 1.0}


def main():
    raw, out, frames = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = list(csv.reader(open(raw)))
    h, u, data = rows[0], rows[1], rows[2:]
    c = {x: i for i, x in enumerate(h)}
    nv = next(i for i, x in enumerate(h) if "Push/Pop_Range" in x)

    def g(r, m):
        x = r[c[m]].replace(",", "")
        return float(x) * UNIT.get(u[c[m]], 1.0) if x not in ("", "n/a") else 0.0
    agg, seen_graph = {}, {}
    for r in data:
        rng = next((t for t in r[nv].replace('"', " ").split() if t.endswith(("_prefix", "_expert"))), None)
        if rng is None:
            import re
            m = re.search(r":(\w+_(?:prefix|expert)):", r[nv])
            rng = m.group(1) if m else None
        if rng is None:
            continue
        if rng.endswith("_expert") and r[c["Kernel Name"]] == "graph":
            seen_graph[rng] = seen_graph.get(rng, 0) + 1
            if seen_graph[rng] <= 2:
                continue
        a = agg.setdefault(rng, {"workloads": 0, "reads": 0.0, "writes": 0.0, "l2_read": 0.0, "ms": 0.0})
        a["workloads"] += 1
        a["reads"] += g(r, "lts__d_sectors_fill_device.sum") + g(r, "lts__d_sectors_fill_sysmem.sum")
        a["writes"] += g(r, "lts__t_sectors_op_write.sum")
        a["l2_read"] += g(r, "lts__t_sectors_op_read.sum")
        a["ms"] += g(r, "gpu__time_duration.sum")
    res = []
    for rng, a in sorted(agg.items()):
        v, st = rng.rsplit("_", 1)
        res.append({"variant": v, "stage": st, "workloads_per_frame": a["workloads"] / frames,
                    "measured_read_GB_per_frame": round(a["reads"] / frames / 1e9, 4),
                    "measured_write_GB_per_frame_upper": round(a["writes"] / frames / 1e9, 4),
                    "measured_total_GB_per_frame": round((a["reads"] + a["writes"]) / frames / 1e9, 4),
                    "L2_read_requests_GB_per_frame": round(a["l2_read"] / frames / 1e9, 4),
                    "profiled_kernel_time_ms_per_frame": round(a["ms"] / frames, 3)})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
