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
import glob
import json
import os
import subprocess
import sys

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
    c = sorted(glob.glob(os.path.join(REPO, "results", f"*_E4i_{m}_best_confirm", "raw", "rows", f"{rung}_N{n}.json")))
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
    os.makedirs(os.path.join(e6, "raw"), exist_ok=True)
    G.wait_for_gate(apply_clocks=True)
    if not os.path.exists(os.path.join(an, "accounting.json")):
        rc = sh([py, os.path.join(HERE, "t006_analytic.py"), "--model", m, "--mode", mode, "--out", an], cwd=cwd,
                log=os.path.join(e6, "raw", "analytic.log"))
        if rc:
            raise SystemExit(f"analytic failed for {m} {mode}")
    G.wait_for_gate(apply_clocks=True)
    if not os.path.exists(os.path.join(e5, "kernels.csv")):
        if sh([py, os.path.join(HERE, "t006_capture.py"), "nsys", "--model", m, "--mode", mode, "--out", e5], cwd=cwd):
            raise SystemExit(f"nsys failed for {m} {mode}")
    info = json.load(open(os.path.join(e5, "raw", "target_info.json")))
    desc = info["rung_desc"]
    parts = ["vision", "vlm_prefix"] if desc["regions"] else []
    if desc["regions"]:
        parts.append("ae_total" if (m.startswith("groot") and desc["ae_compiled"]) else "ae_step[0]")
    if m.startswith("groot") and desc["regions"]:
        parts.append("backbone_lm_head")
    for part in parts:
        tag = part.replace("[", "").replace("]", "")
        if glob.glob(os.path.join(e6, "ncu", f"{tag}_s*_raw.csv")):
            continue
        G.wait_for_gate(apply_clocks=True)
        if sh([py, os.path.join(HERE, "t006_capture.py"), "ncu", "--model", m, "--mode", mode, "--part", part, "--out", e6],
              cwd=cwd):
            print(f"!!! ncu failed for {m} {mode} {part}; recorded in raw/ncu_*.log", flush=True)
    row = t005_row(m, mode)
    sh([os.path.expanduser("~/venvs/libero/bin/python"), os.path.join(HERE, "t006_roofline.py"), "--model", m, "--mode", mode,
        "--nsys", e5, "--ncu", os.path.join(e6, "ncu"), "--analytic", an, "--t005-row", row, "--out", e6])
    cov = json.load(open(os.path.join(e6, "coverage.json")))
    bad = {p: c for p, c in cov.items() if c["name_mismatches"] or c["ncu_kernels"] != c["nsys_kernels"]}
    if bad:
        json.dump(bad, open(os.path.join(e6, "raw", "JOIN_FAILED.json"), "w"), indent=1)
        print(f"!!! {m} {mode}: ncu and nsys kernel lists differ: {bad}", flush=True)
    common = {"task": "T006", "model": m, "rung": f"{mode} = {info['rung']}", "precision": json.dumps(desc["precision_flags"]),
              "compile_mode": "; ".join(desc["compiled"]) or "none (eager)", "variant": "; ".join(desc["preps"]) or "none",
              "num_denoise_steps": info["n_default"], "action_horizon": {"groot_n17": 40, "groot_n16": 16, "groot_n15": 16}.get(m, 50),
              "prefix_tokens": info["tokens"].get("prefix_tokens_P", "see the T004 tokens.json"),
              "suffix_tokens": info["tokens"].get("suffix_tokens_S", "see the T004 tokens.json"), "batch": 1,
              "n_warmup": f"{info['warmup_iters']} (until no new graph)", "hw_constants_source": "results/20261022_E1_peak/hw_constants.json",
              "bench_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "throttled": "n/a: profiled run (overhead); times from nsys and the T005 row", "cuobjdump_version": "n/a",
              "inductor_cache": timed_cache(m, mode) or "n/a: nothing compiled",
              "model_repo": "see the T005 row's provenance", "model_commit": "see the T005 row's provenance",
              "checkpoint": "see the T005 row's provenance", "checkpoint_revision": "see the T005 row's provenance"}
    finalise(e5, dict(common, experiment="E5 nsys capture (T006 step 3)", n_timed="2 inferences in the capture window",
                      command=f"bench/t006_capture.py nsys --model {m} --mode {mode} --out {os.path.relpath(e5, REPO)}",
                      nsys_version=subprocess.run(["nsys", "--version"], capture_output=True, text=True).stdout.strip(),
                      ncu_version="n/a", ncu_set="n/a", ncu_filter="n/a", ncu_metrics="n/a", ncu_sm_frequency_hz="n/a",
                      produced_by={"nsys": "nsys stats", "kernels": "bench/t006_capture.py", "target_info": "bench/profile_target.py"},
                      also_raw=[os.path.relpath(p, e5) for p in glob.glob(os.path.join(e5, "nsys", "*"))] + ["kernels.csv"],
                      uncommitted_raw=[{"path": f"scratch/{os.path.basename(e5)}/{m}_{mode}.nsys-rep", "committed": False,
                                        "reason": "nsys binaries embed the system environment (results/20261022_E0_secrets_test)"}]))
    commit(e5, f"T006: {os.path.basename(e5)} (nsys, every kernel with its region)")
    finalise(e6, dict(common, experiment="E6 roofline per part and per kernel (T006 steps 4-6)", n_timed="1 inference per ncu run",
                      command=[f"bench/t006_analytic.py --model {m} --mode {mode}", f"bench/t006_capture.py ncu --part <{','.join(parts)}>",
                               f"bench/t006_roofline.py --t005-row {os.path.relpath(row, REPO)}"],
                      nsys_version="see the E5 directory", ncu_version="2025.3.0.0", ncu_set="n/a: form D, explicit metric list",
                      ncu_filter="--nvtx --nvtx-include <part>/ --profile-from-start off, one inference",
                      ncu_metrics="raw/ncu_metrics.txt (T003-validated counters)", ncu_sm_frequency_hz="locked by jetson_clocks; --clock-control none",
                      produced_by={"ncu": "ncu --import --csv --page raw", "roofline": "bench/t006_roofline.py"},
                      also_raw=[os.path.relpath(p, e6) for p in glob.glob(os.path.join(e6, "ncu", "*"))]))
    commit(e6, f"T006: {os.path.basename(e6)} (roofline per part and per kernel)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", default="pi0,pi05,smolvla,groot_n17,groot_n16,groot_n15")
    ap.add_argument("--modes", default="best_bf16,eager_fp32")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the result directory names (e.g. _v2)")
    a = ap.parse_args()
    for mode in a.modes.split(","):
        for m in a.models.split(","):
            print(f"##### {m} {mode}", flush=True)
            one(a.date, m, mode, a)


if __name__ == "__main__":
    main()
