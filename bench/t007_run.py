"""T007 C1 driver: ncu `--set full` + the stall breakdown per part (bench/t007_c1.py) for each model x mode, then the
C1 tables (bench/t007_c1_tables.py, when present), meta.json and one commit + push per directory.

Each capture runs exactly the timed configuration: best_bf16 resolves through results/*_E4i_summary/best_bf16.json,
and the profiling process reuses the Inductor / Triton caches of the run that timed it (bench/t006_run.timed_cache).
The parts are those T006's nsys capture exposed (raw/target_info.json of results/<D>_E5_<m>_<mode>_nsys*).

python bench/t007_run.py --date 20261024 --e5-date 20261024 [--models ...] [--modes best_bf16]
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
import t006_run as T6R  # noqa: E402


def parts_for(m, mode, e5_date):
    e5 = sorted(glob.glob(os.path.join(REPO, "results", f"{e5_date}_E5_{m}_{mode}_nsys*")))
    e5 = [d for d in e5 if not os.path.exists(os.path.join(d, "SUPERSEDED.md"))]
    if not e5:
        raise SystemExit(f"no E5 capture for {m} {mode}")
    info = json.load(open(os.path.join(e5[-1], "raw", "target_info.json")))
    desc = info["rung_desc"]
    parts = ["vision", "vlm_prefix"] if desc["regions"] else []
    if desc["regions"]:
        parts.append("ae_total" if (m.startswith("groot") and desc["ae_compiled"]) else "ae_step[0]")
    return parts, info, e5[-1]


def one(D, m, mode, a):
    import gate as G
    py = os.path.expanduser(f"~/venvs/{T6R.VENV[m]}/bin/python")
    cwd = "/tmp" if m.startswith("groot") else REPO
    for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        os.environ.pop(k, None)
    os.environ.update(T6R.timed_cache(m, mode))
    out = os.path.join(REPO, "results", f"{D}_E4g_c1_{m}_{mode}")
    if a.skip_done and os.path.exists(os.path.join(out, "meta.json")):
        print(f"=== {m} {mode}: done, skipped")
        return
    parts, info, e5 = parts_for(m, mode, a.e5_date)
    for part in parts:
        tag = part.replace("[", "").replace("]", "")
        if glob.glob(os.path.join(out, "ncu", f"{tag}_s*_details.csv*")):
            continue
        G.wait_for_gate(apply_clocks=True)
        rc = T6R.sh([py, os.path.join(HERE, "t007_c1.py"), "--model", m, "--mode", mode, "--part", part, "--out", out], cwd=cwd)
        if rc:
            print(f"!!! C1 failed for {m} {mode} {part}; see {out}/raw/ncu_*.log", flush=True)
    T6R.sh([os.path.expanduser("~/venvs/base/bin/python"), os.path.join(HERE, "t007_c1_tables.py"), "--model", m, "--mode", mode,
            "--c1", out, "--nsys", e5, "--t005-row", T6R.t005_row(m, mode)], log=os.path.join(out, "raw", "c1_tables.log"))
    gates = [json.load(open(p)) for p in glob.glob(os.path.join(out, "raw", "c1_gate_*.json"))]
    desc = info["rung_desc"]
    fields = {"task": "T007", "experiment": "E4g C1: ncu --set full + stall breakdown per part", "model": m,
              "rung": f"{mode} = {info['rung']}", "precision": json.dumps(desc["precision_flags"]),
              "compile_mode": "; ".join(desc["compiled"]) or "none (eager)", "variant": "; ".join(desc["preps"]) or "none",
              "num_denoise_steps": info["n_default"], "action_horizon": {"groot_n17": 40, "groot_n16": 16, "groot_n15": 16}.get(m, 50),
              "prefix_tokens": info["tokens"].get("prefix_tokens_P", "see T004"), "suffix_tokens": info["tokens"].get("suffix_tokens_S", "see T004"),
              "batch": 1, "n_warmup": "until no new graph (profile_target.py)", "n_timed": "1 inference per ncu run",
              "hw_constants_source": "results/20261022_E1_peak/hw_constants.json",
              "bench_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "command": f"bench/t007_c1.py --model {m} --mode {mode} --part <{','.join(parts)}>",
              "throttled": "n/a: profiled run", "nsys_version": "n/a", "ncu_version": "2025.3.0.0", "ncu_set": "full",
              "ncu_filter": "--nvtx --nvtx-include <part>/ --profile-from-start off --target-processes application-only, -s/-c windows",
              "ncu_metrics": "the 17 smsp__average_warps_issue_stalled_*_per_issue_active, sm__pipe_tensor*, T003 counters (bench/t007_c1.py)",
              "ncu_sm_frequency_hz": "locked by jetson_clocks; --clock-control none", "cuobjdump_version": "n/a",
              "model_repo": "see the T005 row's provenance", "model_commit": "see the T005 row's provenance",
              "checkpoint": "see the T005 row's provenance", "checkpoint_revision": "see the T005 row's provenance",
              "inductor_cache": T6R.timed_cache(m, mode) or "n/a: nothing compiled", "e5_capture": os.path.relpath(e5, REPO),
              "c1_gates_all_ok": all(g["ok"] for g in gates) if gates else "no capture",
              "produced_by": {"ncu": "ncu --import --csv", "c1_gate": "bench/t007_c1.py", "target_info": "bench/profile_target.py"},
              "also_raw": [os.path.relpath(p, out) for p in glob.glob(os.path.join(out, "ncu", "*"))]}
    T6R.finalise(out, fields)
    T6R.commit(out, f"T007: {os.path.basename(out)} (C1 ncu --set full + stalls per part; gates ok: {fields['c1_gates_all_ok']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--e5-date", required=True)
    ap.add_argument("--models", default="pi0,pi05,smolvla,groot_n17,groot_n16,groot_n15")
    ap.add_argument("--modes", default="best_bf16")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    for mode in a.modes.split(","):
        for m in a.models.split(","):
            print(f"##### C1 {m} {mode}", flush=True)
            one(a.date, m, mode, a)


if __name__ == "__main__":
    main()
