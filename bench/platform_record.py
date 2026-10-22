"""T002 / E0: the platform record at the current power mode (docs/01_measurement_plan.md E0, protocol §1, §4, §8).

Writes into --out:
  platform.json        gate result + every device-state item of protocol §1 and the E0 extras
  gpu_properties.json  torch.cuda.get_device_properties(0), all fields, + ncu device__attribute_* cross-check
  nvpmodel_modes.csv   copied from T001's toolcheck (static file, re-verified against /etc/nvpmodel.conf)
  hw_constants.json    the static half of protocol §8 (no measured peaks, ridge or counter validation)
  raw/                 stdout of every command (raw/cmd_*.txt)
Steps 3 and 4 (clock probe, idle power) are separate commands: bench/gpu_clock_probe.py, bench/power_sample.py.

python bench/platform_record.py --out results/<date>_E0_platform_MAXN --toy-raw results/<T001>/raw/ncu_toy_raw.csv
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402

COMMANDS = {
    "nvpmodel_q_verbose": ["/usr/sbin/nvpmodel", "-q", "--verbose"],
    "nv_tegra_release": ["cat", "/etc/nv_tegra_release"],
    "dpkg_nvidia": ["bash", "-c", "dpkg -l | grep -E 'nvidia-l4t|nvidia-jetpack|cuda|tensorrt|cudnn'"],
    "uname": ["uname", "-a"],
    "driver": ["cat", "/proc/driver/nvidia/version"],
    "nvidia_smi": ["nvidia-smi"],
    "free": ["free", "-h"],
    "free_bytes": ["free", "-b"],
    "swapon": ["bash", "-c", "swapon --show; echo '--- zram'; ls /sys/block | grep zram || echo 'no zram device'"],
    "df": ["df", "-h", "/", "/home"],
    "lsblk": ["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,MODEL"],
    "lscpu": ["lscpu"],
    "cpu_governors_freqs": ["bash", "-c", "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo \"$(basename $c) "
                                          "$(cat $c/cpufreq/scaling_governor) cur=$(cat $c/cpufreq/scaling_cur_freq) "
                                          "min=$(cat $c/cpufreq/scaling_min_freq) max=$(cat $c/cpufreq/scaling_max_freq)\"; done"],
    "devfreq": ["bash", "-c", "for d in /sys/class/devfreq/*; do for f in name governor cur_freq min_freq max_freq "
                              "available_frequencies; do echo \"$d/$f: $(cat $d/$f)\"; done; done"],
    "thermal": ["bash", "-c", "for z in /sys/class/thermal/thermal_zone*; do echo \"$z $(cat $z/type) $(cat $z/temp 2>&1)\"; done"],
    "hwmon": ["bash", "-c", "grep -H . /sys/class/hwmon/hwmon*/name /sys/class/hwmon/hwmon*/*_label /sys/class/hwmon/hwmon*/label 2>/dev/null"],
    "python_versions": ["bash", "-c", "python3 --version; ~/venvs/base/bin/python -c 'import torch,torchvision,triton;"
                                      "print(torch.__version__,torch.version.cuda,torchvision.__version__,triton.__version__)'"],
    "nvcc": [os.path.expanduser("~/opt/bin/nvcc"), "--version"],
}
ROOT_COMMANDS = {
    "jetson_clocks_show": ["jetson_clocks", "--show"],
    "emc_rate": ["cat", "/sys/kernel/debug/bpmp/debug/clk/emc/rate"],
}


def run(args, root=False):
    try:
        r = subprocess.run(([gate.ROOTDO] if root else []) + args, capture_output=True, text=True, timeout=120)
        return r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
    except Exception as exc:
        return f"n/a: {type(exc).__name__}: {exc}"


def gpu_properties(toy_raw):
    import torch
    p = torch.cuda.get_device_properties(0)
    props = {k: getattr(p, k) for k in dir(p) if not k.startswith("_") and isinstance(getattr(p, k), (int, float, str))}
    props["uuid"] = str(p.uuid)
    props["capability"] = list(torch.cuda.get_device_capability())
    props["gpu_arch"] = "sm_%d%d" % torch.cuda.get_device_capability()
    props["max_warps_per_sm"] = props["max_threads_per_multi_processor"] // props["warp_size"]
    ncu = {}
    if toy_raw and os.path.exists(toy_raw):
        rows = list(csv.reader(open(toy_raw)))
        ncu = {k: v for k, v in zip(rows[0], rows[2]) if k.startswith("device__attribute_")}
    return {"torch": props, "ncu_device_attributes_from": toy_raw, "ncu_device_attributes": ncu}


def static_hw_constants(props, st):
    t = props["torch"]
    n = props["ncu_device_attributes"]

    def ncu_int(k):
        v = n.get(k, "")
        try:
            return int(float(v.replace(",", "")))
        except ValueError:
            return "n/a: not in the ncu export"

    return {
        "device": "NVIDIA Jetson AGX Thor Developer Kit",
        "gpu_name": t["name"],
        "compute_capability": f"{t['major']}.{t['minor']}",
        "gpu_arch": t["gpu_arch"],
        "sm_count": t["multi_processor_count"],
        "power_mode": st["power_mode"],
        "gpu_clock_locked_hz": "n/a: filled from clock_probe.csv (median under load) by the T002 finaliser",
        "emc_hz": int(st["emc_hz"]) if str(st["emc_hz"]).isdigit() else st["emc_hz"],
        "l2_bytes": t["L2_cache_size"],
        "smem_per_sm_bytes": t["shared_memory_per_multiprocessor"],
        "smem_per_block_optin_bytes": t["shared_memory_per_block_optin"],
        "max_warps_per_sm": t["max_warps_per_sm"],
        "max_threads_per_sm": t["max_threads_per_multi_processor"],
        "regs_per_sm": t["regs_per_multiprocessor"],
        "warp_size": t["warp_size"],
        "ncu_cross_check": {k: ncu_int(k) for k in ("device__attribute_multiprocessor_count",
                                                     "device__attribute_l2_cache_size",
                                                     "device__attribute_max_shared_memory_per_multiprocessor",
                                                     "device__attribute_max_registers_per_multiprocessor",
                                                     "device__attribute_max_threads_per_multiprocessor",
                                                     "device__attribute_fbp_count",
                                                     "device__attribute_global_memory_bus_width",
                                                     "device__attribute_max_warps_per_multiprocessor",
                                                     "device__attribute_max_blocks_per_multiprocessor",
                                                     "device__attribute_num_schedulers_per_multiprocessor",
                                                     "device__attribute_num_l2s_per_fbp",
                                                     "device__attribute_max_persisting_l2_cache_size",
                                                     "device__attribute_max_gpu_frequency_khz",
                                                     "device__attribute_max_mem_frequency_khz")},
        "ncu_cross_check_note": "max_gpu_frequency_khz (and clock_rate) read 1,048,576 = 2^20, a placeholder on this "
                                "integrated GPU, not a clock; global_memory_bus_width and memory clock read 0 (unified "
                                "LPDDR5X, no discrete frame buffer). The clock comes from devfreq.",
        "memory": {"type": "LPDDR5X", "bus_width_bits": 256, "spec_GBps": 273,
                   "spec_source": "NVIDIA Jetson AGX Thor product page (spec, not measured; docs/00_context.md hint)"},
        "measured_bw_GBps": "n/a: T003 (E1)",
        "measured_tflops": "n/a: T003 (E1)",
        "ridge_flop_per_byte": "n/a: T003 (E1)",
        "spec_tflops": "n/a: T003 records NVIDIA's published figures where they exist",
        "counter_validation": "n/a: T003",
        "source_runs": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--toy-raw", default="results/20261022_E0_toolcheck/raw/ncu_toy_raw.csv")
    ap.add_argument("--modes-csv", default="results/20261022_E0_toolcheck/nvpmodel_modes.csv")
    a = ap.parse_args()
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    before = run(["jetson_clocks", "--show"], root=True)
    open(os.path.join(raw, "cmd_jetson_clocks_show_before_gate.txt"), "w").write(before)
    st = gate.run_gate(apply_clocks=True)
    for name, cmd in COMMANDS.items():
        open(os.path.join(raw, f"cmd_{name}.txt"), "w").write(f"$ {' '.join(cmd)}\n{run(cmd)}")
    for name, cmd in ROOT_COMMANDS.items():
        open(os.path.join(raw, f"cmd_{name}.txt"), "w").write(f"$ env/rootdo.sh {' '.join(cmd)}\n{run(cmd, root=True)}")
    shutil.copy("/etc/nvpmodel.conf", os.path.join(raw, "nvpmodel.conf"))
    conf_sha = hashlib.sha256(open("/etc/nvpmodel.conf", "rb").read()).hexdigest()
    t001_conf = "results/20261022_E0_toolcheck/raw/nvpmodel_p3834_0008.conf"
    same = os.path.exists(t001_conf) and hashlib.sha256(open(t001_conf, "rb").read()).hexdigest() == conf_sha
    shutil.copy(a.modes_csv, os.path.join(a.out, "nvpmodel_modes.csv"))
    props = gpu_properties(a.toy_raw)
    json.dump(props, open(os.path.join(a.out, "gpu_properties.json"), "w"), indent=1)
    platform = {"gate": st, "jetson_clocks_show_before_gate": before,
                "jetson_clocks_already_applied_this_boot": "yes: T001 applied it at 2026-10-22T03:26Z (same boot)",
                "nvpmodel_conf_sha256": conf_sha, "nvpmodel_conf_unchanged_since_T001": same}
    json.dump(platform, open(os.path.join(a.out, "platform.json"), "w"), indent=1)
    json.dump(static_hw_constants(props, st), open(os.path.join(a.out, "hw_constants.json"), "w"), indent=1)
    print(json.dumps({"gate_pass": st["gate_pass"], "checks": st["checks"], "conf_unchanged": same}, indent=1))


if __name__ == "__main__":
    main()
