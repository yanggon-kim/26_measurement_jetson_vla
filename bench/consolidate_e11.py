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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, collect_raw, sh_out, software_versions, write_json,  # noqa: E402
                    write_summary)
from host_segments import MI_HEADER  # noqa: E402

def parts_for(model: str) -> list[str]:
    """The E11 parts that belong to `model`, in the order they are consolidated.

    T011 reshapes this for the models after pi0: **E11b is dropped** and the exact
    transfer measurement moves inside E11c as `xfer_exact.csv`. E11b (xfer_latency.py)
    measures host<->device copies with no model in them at all, so re-running it per
    model would produce the same numbers under a different name and invite someone to
    average them as if they were independent.
    """
    ps = [f"E11a_{model}_hostseg", f"E11c_{model}_handover",
          f"E11d_{model}_eager_prefixsweep",
          f"E11d_{model}_reduce-overhead_prefixsweep",
          f"E11e_{model}_eager_mempressure",
          f"E11e_{model}_reduce-overhead_mempressure",
          f"E11f_{model}_pipelining"]
    if model == "pi0":
        # pi0's E11b ran as its own part before T011 reshaped this, and its rows are
        # committed. Keeping it for pi0 only preserves that history without implying the
        # later models are missing something.
        ps.insert(1, "E11b_xfer")
    return ps
REGION_Q = {"preprocess": "t_preprocess", "vision": "t_vision", "vlm_prefix": "t_vlm_prefix",
            "postprocess": "t_postprocess", "ae_step": "t_ae_step", "ae_total": "t_ae_total",
            "e2e_wall": "t_e2e", "e2e": "t_e2e"}


DATED = re.compile(r"^\d{8}[a-z]?_")


def pick_run(pattern: str) -> Path | None:
    """The newest **dated** result directory matching `pattern`.

    Two rules, both learned the hard way. Take the *newest* match, because a re-measured
    part lives beside the superseded one (§5 forbids overwriting a result directory) and
    the date prefix sorts. But take only directories whose name starts with a date:
    `results/` also holds diagnostic directories named `evidence_…` and `regions_…`, and
    plain lexical order puts those after every date, so `[-1]` silently preferred
    `evidence_E4_pi0_reduce-overhead_N4` over `20260921_E4_pi0_reduce-overhead_N4`.
    """
    cands = [p for p in sorted((REPO / "results").glob(pattern)) if DATED.match(p.name)]
    return cands[-1] if cands else None


def restate(out: Path, src_dir: Path, filename: str) -> str | None:
    """Copy a tidy CSV under raw/restated/<run>/ and return its relative path."""
    src = src_dir / filename
    if not src.exists():
        return None
    dst = out / "raw" / "restated" / src_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / filename)
    return f"raw/restated/{src_dir.name}/{filename}"


def _pick_ps(rows: list[list], which: str):
    """P or S as the PARTS reported it, never asserted.

    pi0's P = 816 is a property of its tokenizer_max_length=48 and empty_cameras=1;
    pi0.5 has 200 and 0. Hardcoding pi0's constant into a consolidated file for another
    model would put a wrong number in the one artefact the research side reads directly.
    """
    i = MI_HEADER.index(which)
    seen = {r[i] for r in rows if r[i] not in ("", None)}
    if len(seen) == 1:
        v = seen.pop()
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    if not seen:
        return None
    return sorted(seen)          # disagreement is reported, not averaged away



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="pi0", help="pi0 | pi05 | smolvla")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    bench_commit = sh_out(f"git -C {REPO} rev-parse HEAD")
    import model_registry as MR                                       # noqa: PLC0415
    spec = MR.get(args.model)
    PARTS = parts_for(args.model)
    rows, provenance, missing = [], [], []

    # --- 1. every part's own rows, with source_file repointed at a restated copy
    for part in PARTS:
        d = pick_run(f"*_{part}")
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
    # E3 and E4 ran for pi0 only (T002/T003). For any other model the equivalent region
    # latencies live in the Part B ladder rows instead, so this section is SKIPPED rather
    # than silently matching pi0's directories and copying pi0's numbers into another
    # model's file -- which is exactly what the glob would have done.
    for tag, pat, style, cm in () if args.model != "pi0" else (
                               ("E3 eager", "*_E3_pi0_eager_N{n}", "eager", ""),
                                ("E4 reduce-overhead",
                                 "*_E4_pi0_reduce-overhead_N{n}", "compiled",
                                 "reduce-overhead")):
        for n_steps in (4, 10):
            d = pick_run(pat.format(n=n_steps))
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
    # pi0 only, for the same reason as section 2: this summary file is pi0's.
    for name, style, cm in (() if args.model != "pi0"
                            else (("20260921_E4_pi0_summary.csv", "", ""),)):
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
        "experiment": "E11 (consolidated)",
        "task": "T010" if args.model == "pi0" else "T011",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.name,
        "model_repo": spec.checkpoint,
        "model_commit": spec.revision,
        "checkpoint": spec.checkpoint,
        "precision": "bfloat16", "compile_mode": "eager and reduce-overhead",
        # P and S are READ from the parts' own rows, never asserted: pi0.5 has
        # tokenizer_max_length=200 and empty_cameras=0 against pi0's 48 and 1, so pi0's
        # P = 816 does not carry over to any other model.
        "batch": 1, "prefix_tokens": _pick_ps(rows, "P"),
        "suffix_tokens": _pick_ps(rows, "S"),
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
         f"{spec.name} `{spec.checkpoint}`"
         + (f" @ `{spec.revision[:8]}`" if spec.revision else "")
         + f", bf16, MAXN + jetson_clocks, P = {_pick_ps(rows, 'P')}, "
         f"S = {_pick_ps(rows, 'S')}. Each row's own run directory holds its gate "
         "record, device state and raw files; this directory is derived and measures nothing "
         "itself."),
        ("Reproduce",
         f"```\npython bench/consolidate_e11.py --model {args.model} "
         f"--outdir {args.outdir}\n"
         f"python docs/reference_scripts/check_model_inputs.py "
         f"{args.outdir}/model_inputs.csv\n```"),
    ], [f"{q}: not measured — see reports/E11_{spec.name}_hybrid_inputs.md"
        for q in absent]
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
