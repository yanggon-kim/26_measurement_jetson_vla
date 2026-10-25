"""T006 driver: for each mode (best_bf16 first, the owner's order) and model, the gate, then
  analytic  bench/t006_analytic.py   -> results/<D>_E6_dims/raw/<m>_<mode>/        (C3 dims + accounting)
  nsys      bench/t006_capture.py    -> results/<D>_E5_<m>_<mode>_nsys/            (E5 capture, every-kernel table)
  ncu       bench/t006_capture.py    -> results/<D>_E6_<m>_<mode>/ncu/             (form D, one part per run)
  roofline  bench/t006_roofline.py   -> results/<D>_E6_<m>_<mode>/                 (join, tables, plot)
then meta.json for both result directories, and one commit + push each.
Parts: vision, vlm_prefix, the expert (ae_step[0] when the per-step regions exist, else ae_total), and GR00T's
backbone_lm_head.

python bench/t006_run.py --date 20261023 [--models pi0,...] [--modes best_bf16,eager_fp32] [--skip-done]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
VENV = {"pi0": "lerobot", "pi05": "lerobot", "smolvla": "lerobot", "groot_n17": "groot17", "groot_n16": "groot16",
        "groot_n15": "groot15"}
LADDER = {"pi0": "E4c_pi0_opt_ladder", "pi05": "E4c_pi05_opt_ladder", "smolvla": "E4c_smolvla_opt_ladder",
          "groot_n17": "E4d_groot_n17_ladder", "groot_n16": "E4d_groot_n16_ladder", "groot_n15": "E4d_groot_n15_ladder"}


def sh(cmd, cwd=REPO, log=None):
    print("+", " ".join(cmd)[:220], flush=True)
    if log:
        with open(log, "w") as f:
            return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=cwd).returncode


def t005_row(m, mode):
    """The T005 row of this configuration: best_bf16 -> its confirmation run (else the search row); eager_fp32 ->
    the ladder."""
    n = 4 if m.startswith("groot") else 10
    if mode == "eager_fp32":
        return glob.glob(os.path.join(REPO, "results", f"*_{LADDER[m]}", "raw", "rows", f"eager_fp32_N{n}.json"))[-1]
    import opt_ladder as OL
    rung, src = OL.resolve_best(m)
    best = json.load(open(src))["models"][m]
    c = sorted(glob.glob(os.path.join(REPO, "results", f"*_E4i_{m}_best_confirm*", "raw", "rows", f"{rung}_N{n}.json")))
    return c[-1] if c else os.path.join(REPO, best["timed_in"], "raw", "rows", f"{rung}_N{n}.json")


def finalise(run, fields):
    py = os.path.expanduser("~/venvs/base/bin/python")
    fp = os.path.join(run, "raw", "meta_fields.json")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    json.dump(fields, open(fp, "w"), indent=1, default=str)
    scan = os.path.join(run, "raw", "secrets_scan.csv")
    files = [p for p in glob.glob(f"{run}/**", recursive=True) if os.path.isfile(p)]
    subprocess.run([py, os.path.join(HERE, "secrets_scan.py"), "--out", scan, "--token-file",
                    os.path.expanduser("~/.cache/huggingface/token")] + files, check=False)
    subprocess.run([py, os.path.join(HERE, "meta.py"), "--run", run, "--fields", fp, "--device-state", "--scan", scan], check=True)


def commit(path, msg):
    subprocess.run(["git", "add", path], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], cwd=REPO)
    subprocess.run(["git", "pull", "--rebase", "--autostash", "-q"], cwd=REPO)
    subprocess.run(["git", "push", "-q"], cwd=REPO)


def timed_cache(m, mode):
    """The Inductor / Triton caches of the run that TIMED this configuration (best_bf16: its confirmation run).
    Autotuning picks kernels by timing them in each process, so a fresh process can compile different kernels (the
    first SmolVLA capture did: 257 of 284 expert kernels differed). Reusing the timed run's caches loads exactly the
    kernels that were timed. eager_fp32 compiles nothing."""
    if mode != "best_bf16":
        return {}
    row = t005_row(m, mode)
    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(row)))
    rung = os.path.basename(row).rsplit("_N", 1)[0]
    base = os.path.join(REPO, "scratch", "t005", "cache", os.path.basename(run_dir), rung)
    if not os.path.isdir(base):
        raise SystemExit(f"no Inductor cache for the timed run of {m} {mode} at {base}")
    return {"TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"), "TRITON_CACHE_DIR": os.path.join(base, "triton")}


def device_snap():
    """Device state around a GPU step: soctherm over-current counters, GPU devfreq clock, GPU temperature."""
    import gate as G
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oc": G.oc_counts(),
            "gpu_cur_hz": G.rd_int(f"{G.DEVFREQ}/cur_freq"), "gpu_C": G.rd("/sys/class/thermal/thermal_zone1/temp")}


def step(run, name, cmd, cwd, log=None):
    """Run one GPU capture step with device snapshots before and after (raw/device_during.json)."""
    p = os.path.join(run, "raw", "device_during.json")
    rec = json.load(open(p)) if os.path.exists(p) else []
    before = device_snap()
    rc = sh(cmd, cwd=cwd, log=log)
    rec.append({"step": name, "before": before, "after": device_snap(), "rc": rc})
    json.dump(rec, open(p, "w"), indent=1)
    return rc


def throttled_of(run):
    """Any OC event or devfreq drop across the run's GPU steps -> True (protocol §5 `throttled`)."""
    import gate as G
    p = os.path.join(run, "raw", "device_during.json")
    if not os.path.exists(p):
        return "unknown: no device_during.json"
    rec = json.load(open(p))
    v = [G.throttle_verdict(r["before"]["oc"], r["after"]["oc"], [r["before"]["gpu_cur_hz"], r["after"]["gpu_cur_hz"]])
         for r in rec]
    t = [x["throttled"] for x in v]
    return {"throttled": any(x is True for x in t) if all(isinstance(x, bool) for x in t) else "unknown: " + str(t),
            "steps": [{"step": r["step"], **x} for r, x in zip(rec, v)], "source": "raw/device_during.json"}


def session_peak(D):
    """T006 configuration: bench/micro_peak.py at the start of each profiling session (short: 1 GiB copy x 3, GEMM n = 8192
    in bf16 / TF32 / fp32 x 3), its value beside the T003 ceilings. results/<D>_E1_session_peak, once per date."""
    out = os.path.join(REPO, "results", f"{D}_E1_session_peak")
    if os.path.exists(os.path.join(out, "meta.json")):
        return os.path.relpath(out, REPO)
    import gate as G
    G.wait_for_gate(apply_clocks=True)
    py = os.path.expanduser("~/venvs/base/bin/python")
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    cmds = [[py, os.path.join(HERE, "micro_peak.py"), "--part", "bw", "--proc-tag", "p1", "--out", out, "--repeats", "3",
             "--bw-gib", "1"],
            [py, os.path.join(HERE, "micro_peak.py"), "--part", "gemm", "--proc-tag", "p1", "--out", out, "--repeats", "3",
             "--gemm-n", "8192", "--dtypes", "bf16,tf32,fp32"]]
    for c in cmds:
        sh(c, log=os.path.join(out, "raw", f"micro_peak_{c[3]}.log"))
    fields = {"task": "T006", "experiment": "session peak check (bench/micro_peak.py, short) at the start of a profiling session",
              "model": "n/a: device microbenchmark", "rung": "n/a: microbenchmark", "precision": "bf16, tf32, fp32 GEMM; bf16 copy",
              "compile_mode": "none", "variant": "short session check (T003 ran the full sweep)", "num_denoise_steps": "n/a: no model",
              "action_horizon": "n/a: no model", "prefix_tokens": "n/a: no model", "suffix_tokens": "n/a: no model", "batch": "n/a: no model",
              "n_warmup": "per micro_peak.py", "n_timed": "3 repeats per case", "hw_constants_source": "results/20261022_E1_peak/hw_constants.json",
              "bench_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "command": [" ".join(c[1:]) for c in cmds], "throttled": "per row (micro_peak.py records OC deltas per call)",
              "nsys_version": "n/a: no nsys", "ncu_version": "n/a: no ncu", "ncu_set": "n/a: no ncu", "ncu_filter": "n/a: no ncu",
              "ncu_metrics": "n/a: no ncu", "ncu_sm_frequency_hz": "n/a: no ncu", "cuobjdump_version": "n/a: no SASS",
              "model_repo": "n/a: no model", "model_commit": "n/a: no model", "checkpoint": "n/a: no model",
              "checkpoint_revision": "n/a: no model", "notes": "Compared with T003's ceilings in the E6 reports; not a ceiling itself.",
              "also_raw": [os.path.relpath(p, out) for p in glob.glob(os.path.join(out, "*.csv"))]}
    finalise(out, fields)
    commit(out, f"T006: {os.path.basename(out)} (session peak check)")
    return os.path.relpath(out, REPO)


def latest_c1(m, mode):
    ds = [d for d in sorted(glob.glob(os.path.join(REPO, "results", f"*_E4g_c1_{m}_{mode}")))
          if os.path.exists(os.path.join(d, "meta.json")) and not os.path.exists(os.path.join(d, "SUPERSEDED.md"))]
    return ds[-1] if ds else ""


def one(D, m, mode, a):
    import gate as G
    py = os.path.expanduser(f"~/venvs/{VENV[m]}/bin/python")
    cwd = "/tmp" if m.startswith("groot") else REPO
    for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        os.environ.pop(k, None)                       # never carry one configuration's cache into the next
    os.environ.update(timed_cache(m, mode))           # inherited by every capture subprocess (rootdo forwards both)
    e5 = os.path.join(REPO, "results", f"{D}_E5_{m}_{mode}_nsys{a.suffix}")
    e6 = os.path.join(REPO, "results", f"{D}_E6_{m}_{mode}{a.suffix}")
    an = os.path.join(REPO, "results", f"{D}_E6_dims", "raw", f"{m}_{mode}")
    if a.skip_done and os.path.exists(os.path.join(e6, "meta.json")):
        print(f"=== {m} {mode}: done, skipped")
        return
    peak = session_peak(D)
    for d in (e5, e6):
        os.makedirs(os.path.join(d, "raw"), exist_ok=True)
    G.wait_for_gate(apply_clocks=True)
    if not os.path.exists(os.path.join(an, "accounting.json")):
        rc = step(e6, "analytic", [py, os.path.join(HERE, "t006_analytic.py"), "--model", m, "--mode", mode, "--out", an], cwd,
                  log=os.path.join(e6, "raw", "analytic.log"))
        if rc:
            raise SystemExit(f"analytic failed for {m} {mode}")
    G.wait_for_gate(apply_clocks=True)
    if not os.path.exists(os.path.join(e5, "kernels.csv")):
        if step(e5, "nsys", [py, os.path.join(HERE, "t006_capture.py"), "nsys", "--model", m, "--mode", mode, "--out", e5], cwd):
            raise SystemExit(f"nsys failed for {m} {mode}")
    if not os.path.exists(os.path.join(e6, "raw", "torch_kernels.csv")):
        G.wait_for_gate(apply_clocks=True)
        if step(e6, "torchprof", [py, os.path.join(HERE, "t006_capture.py"), "torchprof", "--model", m, "--mode", mode, "--out", e6], cwd):
            print(f"!!! torch.profiler linkage failed for {m} {mode}; per-kernel analytic will be n/a", flush=True)
    info = json.load(open(os.path.join(e5, "raw", "target_info.json")))
    desc = info["rung_desc"]
    parts = ["vision", "vlm_prefix"] if desc["regions"] else []
    if desc["regions"]:
        parts.append("ae_total" if (m.startswith("groot") and desc["ae_compiled"]) else "ae_step[0]")
    if m.startswith("groot") and desc["regions"]:
        parts.append("backbone_lm_head")
    for part in parts:
        tag = part.replace("[", "").replace("]", "")
        if not glob.glob(os.path.join(e6, "ncu", f"{tag}_s*_raw.csv*")):
            G.wait_for_gate(apply_clocks=True)
            if step(e6, f"ncu {part}", [py, os.path.join(HERE, "t006_capture.py"), "ncu", "--model", m, "--mode", mode, "--part", part,
                                        "--out", e6], cwd):
                print(f"!!! ncu failed for {m} {mode} {part}; recorded in raw/ncu_*.log", flush=True)
        if not glob.glob(os.path.join(e6, "ncu", f"{tag}_range_raw.csv*")):
            G.wait_for_gate(apply_clocks=True)
            if step(e6, f"ncu range {part}", [py, os.path.join(HERE, "t006_capture.py"), "ncu-range", "--model", m, "--mode", mode,
                                              "--part", part, "--out", e6], cwd):
                print(f"!!! ncu range replay failed for {m} {mode} {part}; recorded in raw/ncu_*_range.log", flush=True)
    row = t005_row(m, mode)
    c1 = latest_c1(m, mode)
    rcmd = [os.path.expanduser("~/venvs/libero/bin/python"), os.path.join(HERE, "t006_roofline.py"), "--model", m, "--mode", mode,
            "--nsys", e5, "--ncu", os.path.join(e6, "ncu"), "--analytic", an, "--t005-row", row, "--out", e6] + (["--c1", c1] if c1 else [])
    sh(rcmd)
    cov = json.load(open(os.path.join(e6, "coverage.json")))
    bad = {p: c for p, c in cov.items() if isinstance(c, dict) and (c["join_none"] or c["coverage_pct"] < 95)}
    if bad:
        json.dump(bad, open(os.path.join(e6, "raw", "JOIN_BELOW_TARGET.json"), "w"), indent=1)
        print(f"!!! {m} {mode}: join below target: { {p: (c['coverage_pct'], c['join_name_only'], c['join_none']) for p, c in bad.items()} }",
              flush=True)
    prov = json.load(open(row)).get("provenance", {})
    clocks = {p: c.get("sm_clock_MHz") for p, c in cov.items() if isinstance(c, dict)}
    common = {"task": "T006", "model": m, "rung": f"{mode} = {info['rung']}", "precision": json.dumps(desc["precision_flags"]),
              "compile_mode": "; ".join(desc["compiled"]) or "none (eager)", "variant": "; ".join(desc["preps"]) or "none",
              "num_denoise_steps": info["n_default"], "action_horizon": {"groot_n17": 40, "groot_n16": 16, "groot_n15": 16}.get(m, 50),
              "prefix_tokens": info["tokens"].get("prefix_tokens_P", "see the T004 tokens.json"),
              "suffix_tokens": info["tokens"].get("suffix_tokens_S", "see the T004 tokens.json"), "batch": 1,
              "n_warmup": f"{info['warmup_iters']} (until no new graph)", "hw_constants_source": "results/20261022_E1_peak/hw_constants.json",
              "session_peak": peak,
              "bench_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "cuobjdump_version": "n/a: no SASS in T006 (T008)",
              "inductor_cache": timed_cache(m, mode) or "n/a: nothing compiled",
              "model_repo": prov.get("model_repo", "n/a: not in the T005 row"), "model_commit": prov.get("model_commit", "n/a: not in the T005 row"),
              "checkpoint": prov.get("checkpoint", "n/a: not in the T005 row"),
              "checkpoint_revision": prov.get("checkpoint_revision", "n/a: not in the T005 row"), "t005_row": os.path.relpath(row, REPO)}
    finalise(e5, dict(common, experiment="E5 nsys capture (T006 step 3)", n_timed="2 inferences in the capture window",
                      command=f"bench/t006_capture.py nsys --model {m} --mode {mode} --out {os.path.relpath(e5, REPO)}",
                      throttled=throttled_of(e5),
                      nsys_version=subprocess.run(["nsys", "--version"], capture_output=True, text=True).stdout.strip(),
                      ncu_version="n/a: nsys capture", ncu_set="n/a: nsys capture", ncu_filter="n/a: nsys capture",
                      ncu_metrics="n/a: nsys capture", ncu_sm_frequency_hz="n/a: nsys capture (GPU devfreq in raw/device_during.json)",
                      notes="nsys timeline of 2 steady-state inferences; kernel durations are the kernel times of the T006 roofline.",
                      produced_by={"nsys": "nsys stats", "kernels": "bench/t006_capture.py", "target_info": "bench/profile_target.py",
                                   "device_during": "bench/t006_run.py", "capture.json": "bench/t006_capture.py"},
                      also_raw=[os.path.relpath(p, e5) for p in glob.glob(os.path.join(e5, "nsys", "*"))] + ["kernels.csv"],
                      uncommitted_raw=[{"path": f"scratch/{os.path.basename(e5)}/{m}_{mode}.nsys-rep", "committed": False,
                                        "reason": "nsys binaries embed the system environment (results/20261022_E0_secrets_test)",
                                        "regenerate": f"bench/t006_capture.py nsys --model {m} --mode {mode} --out <dir>"}]))
    commit(e5, f"T006: {os.path.basename(e5)} (nsys, every kernel with its region)")
    finalise(e6, dict(common, experiment="E6 roofline per part and per kernel (T006 steps 2, 4-6)", n_timed="1 inference per ncu run",
                      command=[f"bench/t006_analytic.py --model {m} --mode {mode} --out {os.path.relpath(an, REPO)}",
                               f"bench/t006_capture.py torchprof --model {m} --mode {mode} --out {os.path.relpath(e6, REPO)}",
                               f"bench/t006_capture.py ncu --part <{','.join(parts)}> --out {os.path.relpath(e6, REPO)}",
                               f"bench/t006_capture.py ncu-range --part <{','.join(parts)}> --out {os.path.relpath(e6, REPO)}",
                               " ".join(os.path.relpath(x, REPO) if x.startswith(REPO) else x for x in rcmd[1:])],
                      throttled=throttled_of(e6),
                      nsys_version="n/a: the nsys capture is " + os.path.relpath(e5, REPO), ncu_version="2025.3.0.0",
                      ncu_set="n/a: form D, explicit metric list (no --set)",
                      ncu_filter="--nvtx --nvtx-include <part>/ --profile-from-start off, one inference; per-kernel replay (caches flushed) "
                                 "and range replay (--replay-mode range --cache-control none)",
                      ncu_metrics="bench/t006_capture.py METRICS / RANGE_METRICS (T003-validated counters + clocks + tensor pipe)",
                      ncu_sm_frequency_hz={p: v for p, v in clocks.items()},
                      notes=("Per-kernel ncu bytes are cold-L2 (caches flushed before each replay): an upper bound; range-replay "
                             "bytes are in-sequence. Kernel times from nsys. Join on name + grid + block (coverage.json). Floors per "
                             "protocol §9 with GEMM activations (C3) and unpadded FLOPs (floor.csv notes). Analytic formulas: GEMM 2·M·N·K "
                             "(M = rows actually multiplied, real S / P; K, N from the weight); attention 4·B·H·L·S·D (QKᵀ and PV; "
                             "FlopCounterMode's formula, registered for flash-attn's custom ops too); convolution per "
                             "FlopCounterMode; elementwise / norm FLOPs not counted (bytes only); bytes = weights + KV / hand-over "
                             "+ GEMM operand activations (inputs read, outputs written)."),
                      produced_by={"ncu": "ncu --import --csv", "roofline": "bench/t006_roofline.py", "torch_kernels": "bench/profile_target.py",
                                   "device_during": "bench/t006_run.py", ".log": "stdout of the step", "target_info": "bench/profile_target.py"},
                      also_raw=[os.path.relpath(p, e6) for p in glob.glob(os.path.join(e6, "ncu", "*"))]))
    commit(e6, f"T006: {os.path.basename(e6)} (roofline per part and per kernel)")


def finalise_dims(D):
    """meta.json + commit for results/<D>_E6_dims (the C3 dims and analytic accounting of every model x mode done so
    far). Re-run after each mode: raw_files is recomputed from what is there."""
    d = os.path.join(REPO, "results", f"{D}_E6_dims")
    done = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "raw", "*_*")) if os.path.isdir(p))
    rows = []                                          # the task's results/<date>_E6_dims/dims.csv: every model x mode
    for sub in done:
        dp = os.path.join(d, "raw", sub, "dims.csv")
        if os.path.exists(dp) and not os.path.exists(os.path.join(d, "raw", sub, "SUPERSEDED.md")):
            rows += list(csv.DictReader(open(dp)))
    if rows:
        with open(os.path.join(d, "dims.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    fields = {"task": "T006", "experiment": "C3 operand dims + analytic FLOPs / bytes per part (T006 steps 1-2)",
              "model": ", ".join(done), "rung": "per subdirectory <model>_<mode>; best_bf16 resolved through *_E4i_summary/best_bf16.json",
              "precision": "the mode's (bench/rungs.py preparations, nothing compiled)", "compile_mode": "none: FlopCounterMode needs eager dispatch",
              "variant": "the mode's preps (bf16 casts, bf16_islands, linear_mpad padded rows counted)",
              "num_denoise_steps": "model default (10 LeRobot, 4 GR00T)", "action_horizon": "model default",
              "prefix_tokens": "measured on a real inference (dims.csv M)", "suffix_tokens": "measured on a real inference (dims.csv M)",
              "batch": 1, "n_warmup": "n/a: no timing", "n_timed": "1 accounting inference", "throttled": "n/a: no timing",
              "hw_constants_source": "results/20261022_E1_peak/hw_constants.json",
              "bench_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "command": [f"bench/t006_analytic.py --model <m> --mode <mode> --out results/{D}_E6_dims/raw/<m>_<mode>"],
              "nsys_version": "n/a", "ncu_version": "n/a", "ncu_set": "n/a", "ncu_filter": "n/a", "ncu_metrics": "n/a",
              "ncu_sm_frequency_hz": "n/a", "cuobjdump_version": "n/a",
              "model_repo": "see the T005 rows", "model_commit": "see the T005 rows", "checkpoint": "see the T005 rows",
              "checkpoint_revision": "see the T005 rows", "also_raw": ["dims.csv"] if rows else [],
              "notes": "dims.csv at the top = every current model x mode's C3 rows (raw/<model>_<mode>/dims.csv merged; superseded "
                       "subdirectories excluded).", "produced_by": {"dims.csv": "bench/t006_analytic.py",
                                                                         "accounting.json": "bench/t006_analytic.py (run_e3.accounting_pass)"}}
    finalise(d, fields)
    commit(d, f"T006: {os.path.basename(d)} (C3 dims + analytic accounting: {', '.join(done)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", default="pi0,pi05,smolvla,groot_n17,groot_n16,groot_n15")
    ap.add_argument("--modes", default="best_bf16,eager_fp32")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the result directory names (e.g. _v2)")
    ap.add_argument("--finalise-dims", action="store_true", help="only write meta.json for <D>_E6_dims and commit it")
    a = ap.parse_args()
    if a.finalise_dims:
        finalise_dims(a.date)
        return
    for mode in a.modes.split(","):
        for m in a.models.split(","):
            print(f"##### {m} {mode}", flush=True)
            one(a.date, m, mode, a)


if __name__ == "__main__":
    main()
