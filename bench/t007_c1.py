"""T007 C1: what each part is bound by. ncu `--set full` plus explicit metrics on every kernel of one part.

Per (model, mode, part): one NVTX-filtered run of bench/profile_target.py (one inference), protocol §3.2 form C:
  --set full                          the full section set (speed of light, memory workload, scheduler, warp state,
                                      occupancy, source counters, ...)
  --metrics <explicit>                the stall breakdown this chip exposes (17 smsp__average_warps_issue_stalled_*_per_issue_active,
                                      env/tool_help/ncu_query_metrics_chip_gb10b.txt), tensor-pipe activity
                                      (sm__pipe_tensor_cycles_active, ..._subpipe_hmma / _imma), and T003's validated
                                      DRAM-byte / FLOP counters, because the sections alone do not carry all of them
                                      (the Orin's C1 lesson)
  --nvtx --nvtx-include <part>/ --profile-from-start off -s/-c windows (never unfiltered)
Exports per window: <tag>_details.csv (long form, every section) and <tag>_raw.csv; every file above 1 MB is
zstd-compressed (protocol §6). Windows are small (--chunk, default 150 kernels) so that a compressed report stays far
below 50 MB; check the sizes after the first capture and lower --chunk if needed.
Gate (ported from reference/orin/bench/c1_capture_ok.py): a capture counts only if its CSV carries all 17 stall
metrics and at least one row per kernel; the result is written to raw/c1_gate_<tag>.json.

python bench/t007_c1.py --model pi0 --mode best_bf16 --part vlm_prefix --out results/<date>_E4g_c1_pi0_best_bf16
"""
import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t006_capture as T6  # noqa: E402

STALLS = [f"smsp__average_warps_issue_stalled_{s}_per_issue_active" for s in (
    "barrier", "branch_resolving", "dispatch_stall", "drain", "lg_throttle", "long_scoreboard", "math_pipe_throttle",
    "membar", "mio_throttle", "misc", "no_instruction", "not_selected", "selected", "short_scoreboard", "sleeping",
    "tex_throttle", "wait")]
TENSOR_PIPE = ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
               "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active",
               "sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_active",
               "sm__inst_executed_pipe_tensor.sum"]
EXTRA = STALLS + TENSOR_PIPE + ["lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_aperture_sysmem_op_write.sum"] + \
    [m for m in T6.METRICS if m.startswith(("sm__ops_path_tensor", "smsp__sass_thread_inst_executed_op"))]


def gate(details_csv, n_kernels):
    """Every requested stall metric present (ncu reports them as <name>.ratio / .pct / .max_rate) and every kernel of the
    raw page present in the details page."""
    if details_csv.endswith(".zst"):
        txt = subprocess.run(["zstd", "-dc", details_csv], capture_output=True, text=True, check=True).stdout
        rows = list(csv.DictReader(io.StringIO(txt)))
    else:
        rows = list(csv.DictReader(open(details_csv)))
    names = {r.get("Metric Name", "") for r in rows}
    names |= {n.rsplit(".", 1)[0] for n in names}
    stalls = [s for s in STALLS if s in names]
    kernels = {r.get("ID") for r in rows}
    ok = len(stalls) == len(STALLS) and len(kernels) >= max(1, n_kernels)
    return {"ok": ok, "rows": len(rows), "stall_metrics_present": len(stalls), "stall_metrics_expected": len(STALLS),
            "kernels_in_details": len(kernels), "kernels_in_raw": n_kernels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for d in ("ncu", "raw"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    alias = a.part.replace("[", "_").replace("]", "")
    tag0 = a.part.replace("[", "").replace("]", "")
    start, k, total, gates = 0, 0, 0, []
    scratch = os.path.join(REPO, "scratch", os.path.basename(out), "ncu_originals")
    while True:
        tag = f"{tag0}_s{start}"
        rep = os.path.join(out, "ncu", tag)
        done = [p for p in (rep + "_details.csv.zst", rep + "_details.csv") if os.path.exists(p)]
        if done and os.path.exists(os.path.join(out, "raw", f"ncu_{tag}.log")) and \
                any(os.path.exists(rep + x) for x in ("_raw.csv", "_raw.csv.zst")):
            # window finished in an earlier run: re-gate it from its saved details page and move on
            n = json.load(open(os.path.join(out, "raw", f"c1_gate_{tag}.json")))["kernels_in_raw"]
            g = gate(done[0], n)
            g["window"] = tag
            gates.append(g)
            json.dump(g, open(os.path.join(out, "raw", f"c1_gate_{tag}.json"), "w"), indent=1)
            total += max(n, 0)
            k += 1
            print(f"{a.part} window -s {start}: done earlier, {n} kernels; gate {g['ok']}", flush=True)
            if n < a.chunk or k >= a.max_windows:
                break
            start += a.chunk
            continue
        for p in glob.glob(rep + ".ncu-rep") + glob.glob(rep + "_*.csv"):
            os.remove(p)                                  # partial output of an interrupted window
        cmd = [os.path.join(REPO, "env", "rootdo.sh"), "ncu", "--clock-control", "none", "--target-processes", "application-only", "--set", "full",
               "--metrics", ",".join(EXTRA), "--nvtx", "--nvtx-include", f"{alias}/", "--profile-from-start", "off",
               "-s", str(start), "-c", str(a.chunk), "-f", "-o", rep] + \
            T6.target_cmd(a, 1, os.path.join(out, "raw", f"target_info_{tag}.json"))
        with open(os.path.join(out, "raw", f"ncu_{tag}.log"), "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=T6.clean_env(),
                                cwd="/tmp" if a.model.startswith("groot") else REPO).returncode
        subprocess.run([os.path.join(REPO, "env", "rootdo.sh"), "chown-repo"], stdout=subprocess.DEVNULL)
        if rc != 0 or not os.path.exists(rep + ".ncu-rep"):
            raise SystemExit(f"ncu exit {rc} at window {start}; see raw/ncu_{tag}.log")
        for page in ("raw", "details"):
            with open(f"{rep}_{page}.csv", "w") as f:
                subprocess.run(["ncu", "--import", rep + ".ncu-rep", "--csv", "--page", page], stdout=f, check=True)
        n = sum(1 for _ in csv.DictReader(open(f"{rep}_raw.csv"))) - 1
        g = gate(f"{rep}_details.csv", n)
        g["window"] = tag
        gates.append(g)
        json.dump(g, open(os.path.join(out, "raw", f"c1_gate_{tag}.json"), "w"), indent=1)
        T6.compress_big([rep + ".ncu-rep", f"{rep}_details.csv", f"{rep}_raw.csv"], scratch)
        total += max(n, 0)
        k += 1
        print(f"{a.part} window -s {start}: {n} kernels; gate {g['ok']} ({g['stall_metrics_present']}/{len(STALLS)} stalls)",
              flush=True)
        if n < a.chunk or k >= a.max_windows:
            break
        start += a.chunk
    json.dump({"part": a.part, "nvtx_include": f"{alias}/", "windows": k, "kernels": total, "chunk": a.chunk,
               "set": "full", "extra_metrics": EXTRA, "gates_all_ok": all(g["ok"] for g in gates)},
              open(os.path.join(out, "raw", f"c1_{tag0}_windows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
