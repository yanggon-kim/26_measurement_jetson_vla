"""T006 captures (E5 nsys, E6 ncu metric-only) for one model x mode, through bench/profile_target.py.

  nsys  --model pi0 --mode best_bf16 --out results/<date>_E5_pi0_best_bf16_nsys
        nsys profile (token variables scrubbed; protocol §3.5) -t cuda,nvtx,osrt,cublas,cudnn, capture range =
        cudaProfilerApi, --cuda-graph-trace=node, 2 steady-state inferences. The .nsys-rep / .sqlite stay in scratch/
        (T001: nsys binaries are NOT CLEAN). Exports: nsys stats CSVs (zstd above 1 MB) and kernels.csv, which lists
        EVERY kernel (launch order, name, grid, block, start, duration, inference, region = the innermost protocol §2
        region whose GPU-projected interval contains it).
  ncu   --model pi0 --mode best_bf16 --part vlm_prefix --out results/<date>_E6_pi0_best_bf16
        Form D (protocol §3.2): only the metrics T003 validated (DRAM bytes from lts sysmem fills / writes x 32,
        tensor FLOPs from sm__ops_path_tensor_src_*, SIMT FLOPs from the sass fadd/fmul/ffma counters, duration, grid,
        block, SM / memory throughput, occupancy), no --set. Filter: the part's NVTX range, one inference, chunked with
        -s / -c windows until every kernel of the range is captured. Run as root through env/rootdo.sh with
        --clock-control none (jetson_clocks holds the clocks); env/rootdo.sh chown-repo afterwards.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = {"pi0": "lerobot", "pi05": "lerobot", "smolvla": "lerobot", "groot_n17": "groot17", "groot_n16": "groot16",
      "groot_n15": "groot15"}
METRICS = (["gpu__time_duration.sum", "launch__grid_size", "launch__block_size", "launch__grid_dim_x", "launch__grid_dim_y",
            "launch__grid_dim_z", "launch__block_dim_x", "launch__registers_per_thread",
            "lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_aperture_sysmem_op_write.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active"]
           + [f"sm__ops_path_tensor_src_{s}.sum" for s in ("bf16_dst_fp32", "fp16_dst_fp16", "fp16_dst_fp32", "tf32_dst_fp32",
                                                             "int8", "fp64", "fp4_dst_fp32", "fp4_fp6_fp8_dst_fp16",
                                                             "fp4_fp6_fp8_dst_fp32")]
           + [f"smsp__sass_thread_inst_executed_op_{p}{o}_pred_on.sum" for p in "fhd" for o in ("add", "mul", "fma")])
REGIONS = ("vision", "vlm_prefix", "backbone_lm_head", "vlm_backbone", "model_get_action", "ae_total")


def clean_env():
    env = {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|SECRET|PASSWORD|APIKEY|API_KEY", k)}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def target_cmd(a, inferences, info):
    return [os.path.expanduser(f"~/venvs/{PY[a.model]}/bin/python"), os.path.join(HERE, "profile_target.py"),
            "--model", a.model, "--rung", a.mode, "--inferences", str(inferences), "--info", info]


def nsys(a):
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out, "nsys"), exist_ok=True)
    scratch = os.path.join(REPO, "scratch", os.path.basename(out))
    os.makedirs(scratch, exist_ok=True)
    rep = os.path.join(scratch, f"{a.model}_{a.mode}")
    cmd = ["nsys", "profile", "-t", "cuda,nvtx,osrt,cublas,cudnn", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", "--cuda-graph-trace=node", "-f", "true", "-o", rep] + \
        target_cmd(a, 2, os.path.join(out, "raw", "target_info.json"))
    t0 = time.time()
    with open(os.path.join(out, "raw", "nsys_stdout.log"), "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=clean_env(),
                            cwd="/tmp" if a.model.startswith("groot") else REPO).returncode
    if rc != 0:
        raise SystemExit(f"nsys exit {rc}; see raw/nsys_stdout.log")
    pre = os.path.join(out, "nsys", f"{a.model}_{a.mode}")
    subprocess.run(["nsys", "stats", "-r", "cuda_gpu_trace,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,"
                    "nvtx_kern_sum,nvtx_gpu_proj_trace,osrt_sum", "-f", "csv", "--force-export=true", "-o", pre, rep + ".nsys-rep"],
                   check=True, stdout=subprocess.DEVNULL, env=clean_env())
    kernels_table(pre, os.path.join(out, "kernels.csv"))
    json.dump({"command": cmd, "wall_s": time.time() - t0, "nsys_rep": rep + ".nsys-rep (scratch/, not committed: T001)"},
              open(os.path.join(out, "raw", "capture.json"), "w"), indent=1)


def kernels_table(pre, path):
    """Every kernel of the capture with its innermost region (by GPU-projected NVTX intervals)."""
    proj = list(csv.DictReader(open(pre + "_nvtx_gpu_proj_trace.csv")))
    kern = [k for k in csv.DictReader(open(pre + "_cuda_gpu_trace.csv")) if k.get("GrdX")]
    iv = []
    for r in proj:
        name = r["Name"].lstrip(":")
        if r["Projected Start (ns)"] and (name in REGIONS or name.startswith(("ae_step[", "ae_dit[", "inference_"))):
            s = int(r["Projected Start (ns)"])
            iv.append((s, s + int(r["Projected Duration (ns)"]), name))
    rows = []
    for i, k in enumerate(sorted(kern, key=lambda k: int(k["Start (ns)"]))):
        s, d = int(k["Start (ns)"]), int(k["Duration (ns)"])
        inside = [(e - b, n) for b, e, n in iv if b <= s and s + d <= e]
        regs = [n for _, n in sorted(inside)]
        inf = next((n for n in regs if n.startswith("inference_")), "")
        part = next((n for n in regs if n.startswith(("ae_step[", "vision", "vlm_prefix", "backbone_lm_head"))), "")
        if not part:
            part = next((n for n in regs if not n.startswith("inference_")), "outside_regions")
        rows.append({"launch_order": i, "inference": inf, "region": part, "name": k["Name"],
                     "grid": f"{k['GrdX']},{k['GrdY']},{k['GrdZ']}", "block": f"{k['BlkX']},{k['BlkY']},{k['BlkZ']}",
                     "start_ns": s, "duration_ns": d, "regions_stack": "|".join(regs)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} kernels -> {path}")


def ncu(a):
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "ncu"), exist_ok=True)
    os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    mfile = os.path.join(out, "raw", "ncu_metrics.txt")
    open(mfile, "w").write(",".join(METRICS))
    alias = a.part.replace("[", "_").replace("]", "")
    chunk, start, n_tot, k = a.chunk, 0, 0, 0
    while True:
        tag = f"{a.part.replace('[', '').replace(']', '')}_s{start}"
        rep = os.path.join(out, "ncu", tag)
        cmd = [os.path.join(REPO, "env", "rootdo.sh"), "ncu", "--clock-control", "none", "--metrics", ",".join(METRICS),
               "--nvtx", "--nvtx-include", f"{alias}/", "--profile-from-start", "off", "-s", str(start), "-c", str(chunk),
               "-f", "-o", rep] + target_cmd(a, 1, os.path.join(out, "raw", f"target_info_{tag}.json"))
        with open(os.path.join(out, "raw", f"ncu_{tag}.log"), "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=clean_env(),
                                cwd="/tmp" if a.model.startswith("groot") else REPO).returncode
        subprocess.run([os.path.join(REPO, "env", "rootdo.sh"), "chown-repo"], stdout=subprocess.DEVNULL)
        if rc != 0 or not os.path.exists(rep + ".ncu-rep"):
            raise SystemExit(f"ncu exit {rc} at window {start}; see raw/ncu_{tag}.log")
        csvp = rep + "_raw.csv"
        with open(csvp, "w") as f:
            subprocess.run(["ncu", "--import", rep + ".ncu-rep", "--csv", "--page", "raw"], stdout=f, check=True)
        n = sum(1 for _ in csv.DictReader(open(csvp))) - 1          # first data row is the units row
        n_tot += max(n, 0)
        k += 1
        print(f"{a.part}: window -s {start} -c {chunk}: {n} kernels", flush=True)
        if n < chunk or k >= a.max_windows:
            break
        start += chunk
    json.dump({"part": a.part, "nvtx_include": f"{alias}/", "windows": k, "kernels": n_tot, "chunk": chunk,
               "metrics": METRICS}, open(os.path.join(out, "raw", f"ncu_{alias}_windows.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["nsys", "ncu"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, help="best_bf16 | eager_fp32 | a rung")
    ap.add_argument("--part", default="vlm_prefix", help="ncu: vision | vlm_prefix | ae_step[0] | ae_total | backbone_lm_head")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--max-windows", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nsys(a) if a.what == "nsys" else ncu(a)


if __name__ == "__main__":
    main()
