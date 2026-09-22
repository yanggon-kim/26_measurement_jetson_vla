"""T010 / E11 step 7 — the consolidated hand-off (`model_inputs.csv`).

Protocol §13.8: one file with every row of E11a–E11f **plus** the E3/E4 region latencies, all
in the same 18-column schema. Rows restated from another run must carry `from=<run dir>` in
`variant` and point at a copy of that run's **tidy CSV** (never its profiler reports) under
`raw/restated/<run dir>/`, because `source_file` is resolved relative to this directory and the
checker verifies the file exists.

    python bench/consolidate_e11.py --outdir results/<date>_E11_pi0_inputs
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, collect_raw, sh_out, software_versions, write_json,  # noqa: E402
                    write_summary)
from host_segments import MI_HEADER  # noqa: E402

PARTS = ["E11a_pi0_hostseg", "E11b_xfer", "E11c_pi0_handover",
         "E11d_pi0_eager_prefixsweep", "E11d_pi0_reduce-overhead_prefixsweep",
         "E11e_pi0_eager_mempressure", "E11e_pi0_reduce-overhead_mempressure",
         "E11f_pi0_pipelining"]
REGION_Q = {"preprocess": "t_preprocess", "vision": "t_vision", "vlm_prefix": "t_vlm_prefix",
            "postprocess": "t_postprocess", "ae_step": "t_ae_step", "ae_total": "t_ae_total",
            "e2e_wall": "t_e2e", "e2e": "t_e2e"}


def restate(out: Path, src_dir: Path, filename: str) -> str | None:
    """Copy a tidy CSV under raw/restated/<run>/ and return its relative path."""
    src = src_dir / filename
    if not src.exists():
        return None
    dst = out / "raw" / "restated" / src_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / filename)
    return f"raw/restated/{src_dir.name}/{filename}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    bench_commit = sh_out(f"git -C {REPO} rev-parse HEAD")
    rows, provenance, missing = [], [], []

    # --- 1. every part's own rows, with source_file repointed at a restated copy
    for part in PARTS:
        d = next(iter(sorted((REPO / "results").glob(f"*_{part}"))), None)
        if d is None or not (d / "model_inputs.csv").exists():
            missing.append(part)
            print(f"  MISSING {part}")
            continue
        n = 0
        for r in csv.DictReader((d / "model_inputs.csv").open()):
            rel = restate(out, d, r["source_file"]) if "/" not in r["source_file"] else None
            if rel is None:
                # the row points at a nested path (or the file is gone); restate the whole
                # model_inputs.csv instead so the pointer still resolves in this directory
                rel = restate(out, d, "model_inputs.csv")
            r["source_file"] = rel
            v = r["variant"]
            r["variant"] = (v + ";" if v else "") + f"from={d.name}"
            rows.append([r[c] for c in MI_HEADER])
            n += 1
        provenance.append({"part": part, "run_dir": d.name, "rows": n})
        print(f"  {part:38s} {n:5d} rows from {d.name}")

    # --- 2. the E3/E4 region latencies (§13.8 copy rule)
    for tag, pat, style, cm in (("E3 eager", "*_E3_pi0_eager_N{n}", "eager", ""),
                                ("E4 reduce-overhead",
                                 "*_E4_pi0_reduce-overhead_N{n}", "compiled",
                                 "reduce-overhead")):
        for n_steps in (4, 10):
            d = next(iter(sorted((REPO / "results").glob(pat.format(n=n_steps)))), None)
            if d is None:
                missing.append(f"{tag} N={n_steps}")
                continue
            rel = restate(out, d, "latency_summary.csv")
            if rel is None:
                missing.append(f"{tag} N={n_steps} latency_summary.csv")
                continue
            cnt = 0
            for r in csv.DictReader((d / "latency_summary.csv").open()):
                q = REGION_Q.get(r.get("region", ""))
                if not q:
                    continue
                var = f"from={d.name};timing_def=instrumented"
                if cm:
                    var += f";compile_mode={cm}"
                for stat, col in (("median", "median_ms"), ("p5", "p5_ms"),
                                  ("p95", "p95_ms")):
                    if not r.get(col):
                        continue
                    rows.append(["pi0", "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
                                 bench_commit, q, "", var, f"{float(r[col]):.9g}", "ms", stat,
                                 r.get("n", "1"), style, "wall_isolated",
                                 "MAXN+jetson_clocks", "bf16", "816", "51", str(n_steps),
                                 rel])
                    cnt += 1
            provenance.append({"part": f"{tag} N={n_steps}", "run_dir": d.name, "rows": cnt})
            print(f"  {tag + f' N={n_steps}':38s} {cnt:5d} rows from {d.name}")

    # --- 3. the slope definition of ms/step, restated from the E3/E4 summaries
    for name, style, cm in (("20260921_E4_pi0_summary.csv", "", ""),):
        f = REPO / "results" / name
        if not f.exists():
            missing.append(name)
            continue
        dst = out / "raw" / "restated"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / name)
        cnt = 0
        for r in csv.DictReader(f.open()):
            slope = r.get("ms_per_step_slope")
            if not slope:
                continue
            conf = r.get("configuration", "")
            st = "eager" if "eager" in conf else "compiled"
            rows.append(["pi0", "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f", bench_commit,
                         "t_ae_step", "", f"from=results;configuration={conf.replace(' ', '_')}",
                         f"{float(slope):.9g}", "ms", "value", r.get("N_points", "1").count(";") + 1,
                         st, "slope", "MAXN+jetson_clocks", "bf16", "816", "51", "",
                         f"raw/restated/{name}"])
            cnt += 1
        provenance.append({"part": "E4 slopes", "run_dir": name, "rows": cnt})
        print(f"  {'E4 slopes':38s} {cnt:5d} rows from {name}")

    with (out / "model_inputs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MI_HEADER)
        w.writerows(rows)
    with (out / "provenance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["part", "run_dir", "rows"])
        w.writeheader()
        w.writerows(provenance)

    quantities = sorted({r[MI_HEADER.index("quantity")] for r in rows})
    ALL_Q = ["t_preprocess", "t_vision", "t_vlm_prefix", "t_postprocess", "t_ae_step",
             "t_ae_total", "t_e2e", "t_host_segment", "n_kernels_host_segment",
             "bytes_host_segment", "flops_host_segment", "t_host_layer", "t_step_extra",
             "t_launch_per_kernel", "t_sync", "xfer_latency", "xfer_latency_fit",
             "kv_bytes_per_layer", "kv_bytes_total", "t_kv_materialise",
             "kv_copied_bytes_per_step", "boundary_bytes_per_step", "prefix_latency_fit",
             "prefix_throughput", "e2e_throughput", "ae_throughput", "prefix_slowdown",
             "host_layer_slowdown", "bg_bw_GBps", "cpu_util", "p_mean_W",
             "gpu_idle_ae_total", "gpu_idle_frac_ae_total", "achieved_bw_GBps",
             "achieved_tflops"]
    absent = [q for q in ALL_Q if q not in quantities]

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11 (consolidated)", "task": "T010",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "lerobot/pi0_libero_finetuned_v044",
        "model_repo": "lerobot/pi0_libero_finetuned_v044",
        "model_commit": "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
        "checkpoint": "lerobot/pi0_libero_finetuned_v044",
        "precision": "bfloat16", "compile_mode": "eager and reduce-overhead",
        "batch": 1, "prefix_tokens": 816, "suffix_tokens": 51,
        "power_mode": "NV Power Mode: MAXN", "bench_commit": bench_commit,
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "torch": sw.get("torch"), "python": sw.get("python"),
        "command": f"python bench/consolidate_e11.py --outdir {args.outdir}",
        "total_rows": len(rows), "quantities_present": quantities,
        "quantities_absent": absent, "parts_missing": missing,
        "provenance": provenance,
        "throttled": False,
        "throttled_evidence": "derived file; every source run carries its own gate record.",
        "raw_files": [],
        "notes": [
            "Every row is restated from a part's own tidy CSV, copied under "
            "raw/restated/<run dir>/ with from=<run dir> in `variant`, per §13.8. Profiler "
            "reports are never copied.",
            "Quantities listed in `quantities_absent` are reported in the task's report with "
            "the reason; §13.8 forbids writing a non-finite value, so an unmeasured quantity "
            "is left out of the file rather than written as blank or NaN.",
        ],
    })
    # Run the §13.8 checker and report what it actually said, rather than asserting it passes.
    chk = subprocess.run([sys.executable,
                          str(REPO / "docs/reference_scripts/check_model_inputs.py"),
                          str(out / "model_inputs.csv")], capture_output=True, text=True)
    (out / "raw" / "check_model_inputs.log").write_text(
        f"$ python docs/reference_scripts/check_model_inputs.py "
        f"{out}/model_inputs.csv\nexit={chk.returncode}\n{chk.stdout}{chk.stderr}")
    check = ("exit 0 — passes" if chk.returncode == 0
             else f"exit {chk.returncode} — FAILS; see `raw/check_model_inputs.log`")
    print(f"  checker: {check}")
    qcount = {}
    for r in rows:
        qcount[r[MI_HEADER.index("quantity")]] = qcount.get(
            r[MI_HEADER.index("quantity")], 0) + 1
    write_summary(out, "E11 — the consolidated hand-off (protocol §13.8)", [
        ("Answer",
         f"`model_inputs.csv` carries **{len(rows)} rows** covering "
         f"**{len(quantities)} of the {len(ALL_Q)}** quantities §13.8 names, drawn from "
         f"{len(provenance)} source runs.\n\nChecker "
         f"(`docs/reference_scripts/check_model_inputs.py`, run in place, never adapted): "
         f"**{check}**"),
        ("Coverage",
         "| quantity | rows |\n|---|---:|\n"
         + "\n".join(f"| {q} | {qcount.get(q, 0)} |" for q in ALL_Q)),
        ("Provenance",
         "| part | run directory | rows |\n|---|---|---:|\n"
         + "\n".join(f"| {p_['part']} | `{p_['run_dir']}` | {p_['rows']} |"
                      for p_ in provenance)
         + "\n\nRows restated from another run carry `from=<run dir>` in `variant` and "
           "`source_file` points at a copy of that run's tidy CSV under "
           "`raw/restated/<run dir>/`, never at a profiler report."),
        ("Conditions",
         "π0 `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc`, bf16, MAXN + jetson_clocks, "
         "P = 816 processed (533 valid), S = 51. Each row's own run directory holds its gate "
         "record, device state and raw files; this directory is derived and measures nothing "
         "itself."),
        ("Reproduce",
         f"```\npython bench/consolidate_e11.py --outdir {args.outdir}\n"
         f"python docs/reference_scripts/check_model_inputs.py "
         f"{args.outdir}/model_inputs.csv\n```"),
    ], [f"{q}: not measured — see reports/E11_pi0_hybrid_inputs.md" for q in absent]
       + [f"part {m} missing" for m in missing])
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}: {len(rows)} rows, {len(quantities)} of {len(ALL_Q)} quantities")
    if absent:
        print(f"  absent: {', '.join(absent)}")
    if missing:
        print(f"  parts missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
