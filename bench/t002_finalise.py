"""T002 finaliser: the derived files of the E0 platform record, from its raw / per-sample files.

Writes (in --run): idle_power_summary.csv, and completes hw_constants.json with the locked clock (clock_probe.csv),
the GPU power limit (raw/power_limits.txt + the OC probes) and a source for every static field (protocol §8).
Definition used everywhere: burst = the first 0.5 s slice of an OC probe (before any OC event); sustained = the median
of the slices from the first slice with an OC event onward.

python bench/t002_finalise.py --run results/20261022_E0_platform_MAXN --oc60 results/20261022_E0_oc_throttle
"""
import argparse
import csv
import json
import os
import re
import statistics as st


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def oc_probe_stats(path):
    rows = list(csv.DictReader(open(path)))
    tf = [float(r["tflops"]) for r in rows]
    first_oc = next((i for i, r in enumerate(rows) if int(r["oc3_delta"]) > 0), len(rows))
    sust = tf[first_oc:]
    return {"source": path, "slices": len(rows), "slice_s": 0.5, "burst_first_slice_tflops": tf[0],
            "first_slice_with_oc_event": first_oc, "sustained_median_tflops": round(st.median(sust), 3),
            "sustained_last_slice_tflops": tf[-1], "slowdown_sustained_vs_burst_pct": round(100 * (1 - st.median(sust) / tf[0]), 1),
            "oc3_events_total": sum(int(r["oc3_delta"]) for r in rows),
            "vdd_gpu_W_median_sustained": st.median(float(r["vdd_gpu_W"]) for r in rows[first_oc:]),
            **({"gpu_C_first": float(rows[0]["gpu_C"]), "gpu_C_last": float(rows[-1]["gpu_C"])} if "gpu_C" in rows[0] else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--oc60", required=True)
    a = ap.parse_args()
    R = a.run
    # idle power summary
    rails = {}
    for r in csv.DictReader(open(f"{R}/idle_power.csv")):
        rails.setdefault(r["rail"], []).append(float(r["mW"]) / 1000)
    roles = {"VIN": "module input = module power [ASSUMPTION: the ina238 channel upstream of the ina3221 rails]",
             "VDD_GPU": "GPU", "VDD_CPU_SOC_MSS": "CPU + SoC + memory subsystem (no separate DRAM rail)",
             "VIN_SYS_5V0": "5 V system"}
    with open(f"{R}/idle_power_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rail", "mean_W", "median_W", "p5_W", "p95_W", "min_W", "max_W", "n", "role"])
        for k, v in rails.items():
            w.writerow([k, round(st.mean(v), 3), round(st.median(v), 3), round(pct(v, .05), 3), round(pct(v, .95), 3),
                        round(min(v), 3), round(max(v), 3), len(v), roles[k]])
    # hw_constants
    hw = json.load(open(f"{R}/hw_constants.json"))
    load = [int(r["cur_freq_hz"]) for r in csv.DictReader(open(f"{R}/clock_probe.csv")) if r["phase"] == "load"]
    hw["gpu_clock_locked_hz"] = {
        "value": int(st.median(load)),
        "source": f"{R}/clock_probe.csv: median of {len(load)} devfreq cur_freq samples (20 Hz, 10 s bf16 4096^3 GEMM "
                  f"loop, jetson_clocks applied); min {min(load)} = max {max(load)}",
        "caveat": "devfreq reports the requested clock; the over-current throttle (gpu_power_limit) is not visible in it",
        "cross_check": "ncu SM frequency 1.5735 GHz (results/20261022_E0_toolcheck); tegrastats GR3D_FREQ 1574 MHz"}
    lim = open(f"{R}/raw/power_limits.txt").read()
    m = re.search(r"VDD_GPU curr_max=(\d+) curr_crit=(\d+)", lim)
    warn, crit = int(m.group(1)), int(m.group(2))
    vbus = st.median(float(r["mV"]) for r in csv.DictReader(open(f"{R}/idle_power.csv")) if r["rail"] == "VDD_GPU") / 1000
    hw["gpu_power_limit"] = {
        "vdd_gpu_current_warning_mA": warn, "vdd_gpu_current_critical_mA": crit, "vdd_gpu_bus_V_idle_median": vbus,
        "vdd_gpu_limit_W_derived": round(warn / 1000 * vbus, 1),
        "oc_throttle": "soctherm oc2 and oc3 have throt_en=1; oc3 fires when VDD_GPU reaches the warning limit",
        "probe_20s": oc_probe_stats(f"{R}/oc_throttle_probe_bf16.csv"),
        "probe_60s_with_temperature": oc_probe_stats(f"{a.oc60}/oc_throttle_probe_bf16_60s.csv"),
        "definition": "burst = first 0.5 s slice (no OC event); sustained = median of the slices from the first OC "
                      "event onward",
        "devfreq_shows_it": False,
        "source": f"{R}/raw/power_limits.txt (ina3221 curr1_max / curr1_crit), {R}/idle_power.csv (bus voltage), "
                  f"the two OC probe CSVs"}
    hw.pop("effect_measured", None)
    gp = f"{R}/gpu_properties.json"
    hw["sources"] = {
        "device": f"{R}/raw/cmd_jetson_clocks_show.txt (Machine:) and /proc/device-tree/model (env/README.md)",
        "gpu_name": gp, "compute_capability": gp, "sm_count": gp, "l2_bytes": gp, "smem_per_sm_bytes": gp,
        "smem_per_block_optin_bytes": gp, "max_warps_per_sm": f"{gp} (max_threads_per_multi_processor / warp_size; "
                                                              "ncu device__attribute_max_warps_per_multiprocessor = 48)",
        "max_threads_per_sm": gp, "regs_per_sm": gp, "warp_size": gp,
        "gpu_arch": "results/20261022_E0_toolcheck/raw/cublaslt_sm_strings.txt and "
                    "results/20261022_E0_addenda/raw/cublaslt_list_text_sm_strings.txt (cuobjdump: sm_110 / sm_110a "
                    "cubins in the cuBLASLt torch loads); torch capability (11, 0) agrees",
        "emc_hz": f"{R}/raw/cmd_emc_rate.txt", "power_mode": f"{R}/raw/cmd_nvpmodel_q_verbose.txt",
        "ncu_cross_check": "results/20261022_E0_toolcheck/raw/ncu_toy_raw.csv"}
    hw["memory"] = {"type": "LPDDR5X (spec, not readable on the device)", "bus_width_bits": "256 (spec; ncu "
                    "device__attribute_global_memory_bus_width reads 0 on this integrated GPU)", "spec_GBps": 273,
                    "spec_source": "NVIDIA Jetson AGX Thor product page, as quoted in docs/00_context.md (a hint, "
                                   "not verified here); measured bandwidth: T003"}
    hw["spec_tflops"] = "n/a: deferred to T003, which records NVIDIA's published per-precision figures with their sources"
    hw["source_runs"] = [R, a.oc60]
    json.dump(hw, open(f"{R}/hw_constants.json", "w"), indent=1)
    print(json.dumps(hw["gpu_power_limit"], indent=1))


if __name__ == "__main__":
    main()
