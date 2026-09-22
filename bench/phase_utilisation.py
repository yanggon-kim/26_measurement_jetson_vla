"""T004 — per-phase SM utilisation and time share, by the method of arXiv:2604.24447 §4.

That paper's Figure 5 reports, for pi0 on an AGX Orin, SM utilisation of 69.4 % (vision),
97.7 % (VLM backbone) and 22.3 % (Action Expert) against a time split of 12 / 28 / 59 %,
and concludes that the phase dominating latency has the worst utilisation. This reproduces
the measurement on our device so the comparison is like-for-like rather than quoted.

Method. Nsight Systems samples GPU hardware metrics on a timer (`--gpu-metrics-devices`,
set `ga10b` for the Orin's Ampere iGPU) while `bench/regions.py` pushes NVTX ranges around
`vision`, `vlm_prefix` and each `ae_step[i]`. Both land in the .nsys-rep; exporting the
SQLite lets each metric sample be assigned to the phase whose range contains its timestamp.

Two device-specific notes, both found by trying:
  * GPU metrics need root here. As an ordinary user `--gpu-metrics-devices=help` answers
    "Orin - Can't load CUDA driver"; under sudo it lists "0: Orin".
  * There is no `nsys stats` report for GPU metrics, so the samples are read from the
    SQLite directly. Table and column names differ between nsys versions, so they are
    discovered at run time rather than hard-coded.

The .sqlite and .nsys-rep are NOT committed (protocol §3.5: they embed the process
environment, including a session token). The derived CSVs are.

    sudo -n /bin/sh -c '... python bench/phase_utilisation.py --outdir <dir> --steps 10'
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, collect_raw, sh_out, software_versions, summarize, write_json  # noqa: E402

PHASES = ("vision", "vlm_prefix", "ae_step")


def discover(con: sqlite3.Connection) -> tuple[str, str, str]:
    """(metrics table, timestamp column, SM-active column) for this nsys version."""
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    cand = [t for t in tables if "GPU_METRIC" in t.upper()]
    if not cand:
        raise SystemExit(f"no GPU metrics table in the export; tables: {tables[:40]}")
    tbl = cand[0]
    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{tbl}')").fetchall()]
    ts = next((c for c in cols if c.lower() in ("timestamp", "start", "rawtimestamp")), None)
    return tbl, ts, ",".join(cols)


def metric_names(con: sqlite3.Connection) -> dict:
    """nsys stores metric ids in a side table; map id -> human name where it exists."""
    out = {}
    for t in ("TARGET_INFO_GPU_METRICS", "GENERIC_EVENT_TYPES", "StringIds"):
        try:
            for r in con.execute(f"SELECT * FROM {t}").fetchall():
                pass
        except sqlite3.Error:
            continue
    try:
        for mid, name in con.execute(
                "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS").fetchall():
            out[mid] = name
    except sqlite3.Error:
        pass
    return out


def nvtx_spans(con: sqlite3.Connection) -> list[tuple[str, int, int]]:
    """(phase, start_ns, end_ns) for every NVTX push/pop range we care about."""
    try:
        rows = con.execute(
            "SELECT s.value, e.start, e.end FROM NVTX_EVENTS e "
            "JOIN StringIds s ON e.textId = s.id WHERE e.end IS NOT NULL").fetchall()
    except sqlite3.Error:
        rows = con.execute(
            "SELECT text, start, end FROM NVTX_EVENTS WHERE end IS NOT NULL").fetchall()
    out = []
    for name, a, b in rows:
        n = (name or "").lstrip(":")
        for p in PHASES:
            if n == p or n.startswith(p + "["):
                out.append((p, a, b))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--inferences", type=int, default=2)
    ap.add_argument("--mode", default="eager")
    ap.add_argument("--freq", type=int, default=10000, help="GPU metrics sampling Hz")
    ap.add_argument("--rep", default=None, help="analyse an existing .nsys-rep instead")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "nsys").mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_N{args.steps}_gpumetrics"
    rep = Path(args.rep) if args.rep else Path("scratch/T004") / f"{tag}.nsys-rep"
    rep.parent.mkdir(parents=True, exist_ok=True)

    if not args.rep:
        cmd = ["nsys", "profile", "-t", "cuda,nvtx",
               "--gpu-metrics-devices=0", "--gpu-metrics-set=ga10b",
               f"--gpu-metrics-frequency={args.freq}",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "-f", "true", "-o", str(rep.with_suffix("")),
               sys.executable, str(REPO / "bench" / "nsys_target.py"),
               "--steps", str(args.steps), "--inferences", str(args.inferences)]
        if args.mode != "eager":
            cmd += ["--compile-mode", args.mode]
        print("capture:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        (out / "raw" / f"{tag}_capture.log").write_text(r.stdout + r.stderr)
        print(f"nsys exit={r.returncode}", flush=True)

    sq = rep.with_suffix(".sqlite")
    if not sq.exists():
        r = subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite",
                            "true", "-o", str(sq), str(rep)],
                           capture_output=True, text=True, timeout=1800)
        print(f"export exit={r.returncode} {r.stderr[-200:] if r.returncode else ''}",
              flush=True)
    con = sqlite3.connect(str(sq))
    tbl, ts, cols = discover(con)
    print(f"metrics table={tbl} ts={ts}\n  columns: {cols}", flush=True)
    names = metric_names(con)
    print(f"  metric ids: {json.dumps(names)[:400]}", flush=True)

    spans = nvtx_spans(con)
    by_phase = {}
    for p, a, b in spans:
        by_phase.setdefault(p, []).append((a, b))
    print("  NVTX spans: " + ", ".join(f"{k}x{len(v)}" for k, v in by_phase.items()),
          flush=True)
    write_json(out / "raw" / f"{tbl}_schema.json",
               {"table": tbl, "timestamp_column": ts, "columns": cols,
                "metric_ids": names,
                "nvtx_spans": {k: len(v) for k, v in by_phase.items()}})
    con.close()
    print("\nSchema discovered. Per-phase aggregation runs once the column names above "
          "are known — rerun with --rep to analyse without recapturing.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
