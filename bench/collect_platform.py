"""E0 / T001 — the platform record (protocol §1 + measurement plan E0).

Writes into one result directory:
  platform.json            everything protocol §1 asks to record, verbatim command output
  nvpmodel_modes.csv       the modes of this board with their clock caps
  tegrastats_rails.csv     which §4 rails this carrier board exposes, and where
  raw/tegrastats_idle.log  a 60 s idle log in the current power mode
  power_idle.csv           tidy (t_s, rail, mW) from that log
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (device_state, ina3221_rails, parse_tegrastats, sh,  # noqa: E402
                    software_versions, write_json)

# the rails protocol §4 asks for
WANTED_RAILS = ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "VDDQ_VDD2_1V8AO"]


def nvpmodel_modes(conf: str = "/etc/nvpmodel.conf") -> list[dict]:
    """Parse the mode table with its clock caps, so E7 can be planned from it."""
    text = Path(conf).read_text()
    modes, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        m = re.match(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", line)
        if m:
            cur = {"id": int(m.group(1)), "name": m.group(2), "params": {}}
            modes.append(cur)
            continue
        if line.startswith("<"):
            cur = None
            continue
        if cur is not None and line:
            parts = line.split()
            if len(parts) >= 3:
                cur["params"][f"{parts[0]}.{parts[1]}"] = parts[2]
    out = []
    for m in modes:
        p = m["params"]
        out.append({
            "id": m["id"],
            "name": m["name"],
            "cpu_cores_online": sum(1 for k, v in p.items()
                                    if k.startswith("CPU_ONLINE") and v == "1"),
            "tpc_pg_mask": p.get("TPC_POWER_GATING.TPC_PG_MASK"),
            "gpu_max_freq_hz": p.get("GPU.MAX_FREQ"),
            "gpu_min_freq_hz": p.get("GPU.MIN_FREQ"),
            "emc_max_freq_hz": p.get("EMC.MAX_FREQ"),
            "cpu_a78_0_max_freq_hz": p.get("CPU_A78_0.MAX_FREQ"),
            "note": "-1 means uncapped; TPC_PG_MASK is a bitmask of power-gated TPCs "
                    "(2 SMs per TPC on Orin)",
        })
    return out


def tegrastats_rail_table(sample_lines: list[str]) -> list[dict]:
    """Which of the §4 rails tegrastats prints, and which only exist in sysfs."""
    printed = set()
    for ln in sample_lines:
        printed |= set(parse_tegrastats(ln)["rails_mW"].keys())
    sysfs = ina3221_rails()
    rows = []
    for rail in sorted(set(WANTED_RAILS) | printed | set(sysfs)):
        rows.append({
            "rail": rail,
            "wanted_by_protocol_s4": rail in WANTED_RAILS,
            "in_tegrastats": rail in printed,
            "in_sysfs_ina3221": rail in sysfs,
            "sysfs_label_path": sysfs.get(rail, {}).get("label_path", ""),
            "example_mW": sysfs.get(rail, {}).get("mW", ""),
        })
    return rows


def tegrastats_idle_log(out_log: Path, seconds: int, interval_ms: int) -> dict:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"timeout {seconds + 2} tegrastats --interval {interval_ms} "
           f"--logfile {out_log}")
    r = sh(cmd, timeout=seconds + 20)
    r["note"] = ("idle baseline in the power mode recorded in platform.json; "
                 "tegrastats needs no sudo on this board")
    return r


def tidy_power(log: Path, out_csv: Path, interval_ms: int) -> int:
    rows = []
    for i, line in enumerate(log.read_text().splitlines()):
        if not line.strip():
            continue
        d = parse_tegrastats(line)
        t = i * interval_ms / 1000.0
        for rail, mw in d["rails_mW"].items():
            rows.append({"t_s": round(t, 3), "rail": rail, "mW": mw})
        if "gr3d_pct" in d:
            rows.append({"t_s": round(t, 3), "rail": "GR3D_FREQ_pct_crosscheck",
                         "mW": d["gr3d_pct"]})
        if "tj_c" in d:
            rows.append({"t_s": round(t, 3), "rail": "tj_degC", "mW": d["tj_c"]})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_s", "rail", "mW"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sudo", default="sudo -n")
    ap.add_argument("--idle-seconds", type=int, default=60)
    ap.add_argument("--idle-interval-ms", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    print("collecting device state ...", flush=True)
    plat = {"device_state": device_state(args.sudo),
            "software": software_versions(),
            "collected_by": "bench/collect_platform.py",
            "argv": sys.argv}

    print("parsing nvpmodel modes ...", flush=True)
    modes = nvpmodel_modes()
    with (out / "nvpmodel_modes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(modes[0].keys()))
        w.writeheader()
        w.writerows(modes)
    plat["nvpmodel_modes"] = modes

    print(f"logging {args.idle_seconds} s of idle tegrastats ...", flush=True)
    log = out / "raw" / "tegrastats_idle.log"
    plat["tegrastats_idle_run"] = tegrastats_idle_log(
        log, args.idle_seconds, args.idle_interval_ms)
    n = tidy_power(log, out / "power_idle.csv", args.idle_interval_ms)
    plat["power_idle_rows"] = n

    rails = tegrastats_rail_table(log.read_text().splitlines()[:5])
    with (out / "tegrastats_rails.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rails[0].keys()))
        w.writeheader()
        w.writerows(rails)
    plat["tegrastats_rails"] = rails

    write_json(out / "platform.json", plat)
    print(f"wrote {out/'platform.json'}  ({n} power rows, "
          f"{len(log.read_text().splitlines())} tegrastats lines)")
    for r in rails:
        print(f"  rail {r['rail']:18s} protocol={r['wanted_by_protocol_s4']!s:5s} "
              f"tegrastats={r['in_tegrastats']!s:5s} sysfs={r['in_sysfs_ina3221']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
