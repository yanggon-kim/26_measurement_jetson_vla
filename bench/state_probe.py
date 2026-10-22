"""T003 follow-up: a hidden performance state of the memory path after sustained heavy GPU load.

Observed: the π0 expert's weight-streaming GEMMs (gate_proj, S = 51) take ≈ 49 µs per call normally but ≈ 34 µs when
measured right after ≥ 30 s of back-to-back bf16 GEMMs, while every clock in /sys/kernel/debug/clk/clk_summary and the
BPMP emc / mcf rates stay the same. This probe tracks the state over time:
  phase idle (--idle-s), phase load (--load-s, back-to-back bf16 4096^3 A @ B), phase recovery (--recover-s).
Once per --period-s it measures a cold weight stream: F.linear(x[51, 1024], W_i) over 18 distinct W_i [4096, 1024]
(144 MiB, > 4x L2), 3 passes, µs per call by CUDA events; during the load phase the probe is interleaved between
load slices. Logged with it: VDD_GPU, VDD_CPU_SOC_MSS, VIN power, soctherm OC deltas, temperatures, devfreq.

python bench/state_probe.py --out results/<run>/state_probe.csv
"""
import argparse
import csv
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402


def rails():
    out = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        name = open(f"{h}/name").read().strip()
        if name == "ina3221":
            for n in "123":
                lab = open(f"{h}/in{n}_label").read().strip()
                out[lab] = int(open(f"{h}/in{n}_input").read()) * int(open(f"{h}/curr{n}_input").read()) / 1e6
        elif name == "ina238":
            out["VIN"] = int(open(f"{h}/power1_input").read()) / 1e6
    return out


def zone(name):
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if open(f"{z}/type").read().strip() == name:
            try:
                return int(open(f"{z}/temp").read()) / 1e3
            except OSError:
                return float("nan")
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-s", type=float, default=20)
    ap.add_argument("--load-s", type=float, default=30)
    ap.add_argument("--recover-s", type=float, default=120)
    ap.add_argument("--period-s", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g0 = gate.run_gate()
    if not g0["gate_pass"]:
        raise SystemExit(f"gate failed {g0['checks']}")
    ws = [torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(18)]
    x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def probe():
        for w in ws:
            F.linear(x, w)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(3):
            for w in ws:
                F.linear(x, w)
        e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3 / (3 * len(ws))

    rows = []
    oc_prev = gate.oc_counts()
    t0 = time.perf_counter()
    phases = [("idle", a.idle_s), ("load", a.load_s), ("recovery", a.recover_s)]
    for phase, dur in phases:
        tp = time.perf_counter()
        while time.perf_counter() - tp < dur:
            tn = time.perf_counter()
            if phase == "load":
                while time.perf_counter() - tn < a.period_s:
                    for _ in range(20):
                        A @ B
                    torch.cuda.synchronize()
            us = probe()
            oc = gate.oc_counts()
            r = rails()
            rows.append({"t_s": round(time.perf_counter() - t0, 3), "phase": phase, "probe_us_per_call": round(us, 3),
                         "weight_GBps": round(4096 * 1024 * 2 / (us * 1e-6) / 1e9, 1),
                         "vdd_gpu_W": round(r.get("VDD_GPU", 0), 2), "vdd_cpu_soc_mss_W": round(r.get("VDD_CPU_SOC_MSS", 0), 2),
                         "vin_W": round(r.get("VIN", 0), 2), "oc3_delta": oc["oc3"] - oc_prev["oc3"],
                         "oc2_delta": oc["oc2"] - oc_prev["oc2"], "gpu_C": zone("gpu-thermal"), "tj_C": zone("tj-thermal"),
                         "devfreq_hz": int(open(f"{gate.DEVFREQ}/cur_freq").read())})
            oc_prev = oc
            if phase != "load":
                time.sleep(max(0.0, a.period_s - (time.perf_counter() - tn)))
            print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
