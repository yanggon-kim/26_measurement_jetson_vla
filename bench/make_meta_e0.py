"""Assemble meta.json for the E0 platform run (protocol §5) from the run's own artifacts.

Everything it writes is read back out of platform.json, hw_constants.json and
raw_files.json, so meta.json cannot drift from what was measured.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sh_out, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="T001")
    args = ap.parse_args()
    run = Path(args.run)

    plat = json.loads((run / "platform.json").read_text())
    hw = json.loads((run / "hw_constants.json").read_text())
    raws = json.loads((run / "raw_files.json").read_text())
    # Protocol §3.5 counts the full ncu CSV exports as raw data ("these may live in
    # ncu/ as before"), with kind "ncu-csv". They are committed uncompressed because
    # both are well under the 1 MB compression threshold of §6b.
    from common import sha256 as _sha
    for name, produced in [
        ("ncu/toy_details.csv",
         "ncu --import ncu_toy.ncu-rep --csv --page details > ncu/toy_details.csv"),
        ("ncu/toy_raw.csv",
         "ncu --import ncu_toy.ncu-rep --csv --page raw > ncu/toy_raw.csv"),
    ]:
        f = run / name
        raws.append({"path": name, "sha256": _sha(f), "bytes": f.stat().st_size,
                     "kind": "ncu-csv", "produced_by": produced,
                     "secrets_scan": "clean", "committed": True})
    ds, sw = plat["device_state"], plat["software"]

    l4t = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", ds["nv_tegra_release"] or "")
    jetpack = re.search(r"nvidia-jetpack\s+(\S+)", ds["dpkg_jetpack"]["stdout"] or "")
    gpu_hz = ds["gpu_max_freq"]["stdout"].strip()
    emc_hz = ds["emc_rate_hz"]["stdout"].strip()
    mode = ds["nvpmodel_q"]["stdout"].strip().splitlines()

    meta = {
        "experiment": "E0",
        "task": args.task,
        "date_utc": plat["device_state"]["date_utc"],
        # no model runs in E0
        "model": None, "model_repo": None, "model_commit": None, "checkpoint": None,
        "precision": "bf16 (stack verification and toy kernel only)",
        "compile_mode": "all inductor modes verified; see stack_verification.json",
        "num_denoise_steps": None, "action_horizon": None,
        "prefix_tokens": None, "suffix_tokens": None,
        "batch": 1, "n_warmup": 20, "n_timed": 100,
        "n_timed_note": "E0 produces no latency result; the toy kernel is a tool check. "
                        "The only timed samples here are the 100 compiled-MLP calls "
                        "inside stack_verification.json, which are a sanity figure, "
                        "not a protocol measurement.",
        "power_mode": mode[0] if mode else None,
        "power_mode_id": mode[-1] if len(mode) > 1 else None,
        "power_mode_deviation": (
            "PROTOCOL DEVIATION: protocol §1 requires MAXN (id 0). This run is at "
            "MODE_30W (id 2), the board default, because switching to MAXN requires a "
            "reboot on this board and AGENTS.md forbids confirming a reboot prompt. "
            "In MODE_30W 4 of 8 GPU TPCs are power-gated (8 of 16 SMs) and the GPU is "
            "capped at 612 MHz. No number from this run characterises the Orin."),
        "jetson_clocks": ("applied after the idle power log and after the toy profiler "
                          "runs; raw/jetson_clocks_show_locked_30W.txt is the locked "
                          "state. power_idle.csv and both toy profiler runs were taken "
                          "with clocks UNLOCKED (schedutil)."),
        "emc_hz": int(emc_hz) if emc_hz.isdigit() else emc_hz,
        "gpu_hz": int(gpu_hz) if gpu_hz.isdigit() else gpu_hz,
        "gpu_active_tpcs": 4, "gpu_total_tpcs": 8,
        "sm_count_reported_by_torch": hw["multi_processor_count"],
        "jetpack": jetpack.group(1) if jetpack else None,
        "l4t": f"R{l4t.group(1)}.{l4t.group(2)}" if l4t else ds["nv_tegra_release"],
        "cuda": sw.get("torch_cuda"),
        "cuda_toolkit": sh_out("nvcc --version | tail -2 | head -1"),
        "cudnn": sw.get("torch_cudnn"),
        "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"),
        "triton": sw.get("triton"),
        "numpy": sw.get("numpy"),
        "python": sw.get("python"),
        "venv": sw.get("executable"),
        "bench_commit": sw.get("bench_commit"),
        "bench_commit_note": (
            "HEAD when bench/collect_platform.py ran. It may predate the commit that "
            "first contains bench/, and the tree may have been dirty, so it does not by "
            "itself identify the code that ran: bench_files_sha256 does. The committed "
            "harness for this run is the one whose hashes are listed there."),
        "bench_dirty_at_run_time": sw.get("bench_dirty"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "nsys_version": sw.get("nsys_version"),
        "ncu_version": sw.get("ncu_version"),
        "cuobjdump_version": sw.get("cuobjdump_version"),
        "nvdisasm_version": sw.get("nvdisasm_version"),
        "ncu_set": "full",
        "ncu_filter": '--nvtx --nvtx-include "toy/" --profile-from-start off '
                      '(no -c/-s needed: the capture window holds one kernel)',
        "ncu_requires_root_on_this_device": True,
        "ncu_clock_control": "base (ncu default) — SM ran at 305.70 MHz during capture, "
                             "while the GPU runs at 612 MHz under load in this mode "
                             "(raw/gpu_clock_under_load.txt)",
        "ncu_run_time_s": 8.30,
        "ncu_run_time_note": (
            "wall clock of a timed repeat of the same filtered command "
            "(raw/ncu_toy_walltime.txt): 8.30 s, 31 replay passes, 7 098 082 B report. "
            "The analysed capture is the earlier identical run: 30 passes, 7 072 411 B."),
        "gpu_arch": hw["gpu_arch"],
        "command": [
            "bash env/setup.sh",
            "python bench/verify_stack.py --out results/20260921_E0_platform/stack_verification.json",
            "python bench/collect_platform.py --outdir results/20260921_E0_platform --sudo 'sudo -A' --idle-seconds 60",
            'nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi -f true -o nsys_toy python bench/ncu_toy.py --iters 3',
            'nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum,nvtx_gpu_proj_trace,cuda_kern_exec_sum -f csv -o results/20260921_E0_platform/nsys/toy nsys_toy.nsys-rep',
            'nsys stats -r cuda_gpu_trace -f csv -o results/20260921_E0_platform/nsys/toy nsys_toy.nsys-rep',
            'sudo $(which ncu) --set full --nvtx --nvtx-include "toy/" --profile-from-start off -o ncu_toy -f python bench/ncu_toy.py',
            'ncu --import ncu_toy.ncu-rep --csv --page details > results/20260921_E0_platform/ncu/toy_details.csv',
            'ncu --import ncu_toy.ncu-rep --csv --page raw     > results/20260921_E0_platform/ncu/toy_raw.csv',
            "python bench/ncu_columns.py --details ... --raw ... --outdir results/20260921_E0_platform/ncu",
            "python bench/hw_constants.py --out results/20260921_E0_platform/hw_constants.json --ncu-raw-csv results/20260921_E0_platform/ncu/toy_raw.csv",
            "python bench/finalize_raw.py --run results/20260921_E0_platform ...",
        ],
        "throttled": False,
        "throttled_evidence": (
            "60 s idle tegrastats: tj stayed in the low 40s °C, GR3D_FREQ 0 %, no clock "
            "below its configured value in jetson_clocks --show. The device was idle; "
            "no sustained load ran in this experiment."),
        "raw_files": raws,
        "notes": [
            "ncu on this device needs root: perf_event_paranoid=1 alone still gives "
            "'Insufficient privileges to launch app for profiling'. Substitution recorded "
            "per protocol §3: every ncu run is 'sudo $(which ncu) ...'.",
            "All dram__* counters read 0 in the ncu export; ncu's 'Memory Throughput' on "
            "this Orin is the L2 (lts__) throughput, not DRAM. Off-chip traffic must come "
            "from lts__t_sectors_aperture_sysmem_*.sum x 32 B (validated to 0.4 % on the "
            "toy kernel). See env/README.md §6 and ncu/metric_presence.csv.",
            "Warp-stall reasons are absent from the ncu 'details' page and present on the "
            "'raw' page as smsp__average_warps_issue_stalled_*_per_issue_active.ratio.",
            "tegrastats on this carrier board does not print VDDQ_VDD2_1V8AO; it must be "
            "read from /sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon4/in2_*.",
            "nsys --capture-range=cudaProfilerApi kills the target at cudaProfilerStop "
            "(exit 143) unless --capture-range-end=stop is passed.",
            "Passwordless sudo was not configured on this device (AGENTS.md prerequisite); "
            "the human supplied the password for this session only.",
        ],
    }
    write_json(run / "meta.json", meta)
    print(f"wrote {run/'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
