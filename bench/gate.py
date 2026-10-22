"""Protocol §1 device gate for this Thor, and the device-state record every run needs.

Checks (all must pass for a measured or profiled run):
  1. power mode is the maximum-performance mode recorded in env/README.md (MAXN, id 0)
  2. SM count equals the count T001 recorded in that mode (20)
  3. jetson_clocks applied in this boot: GPU devfreq min == max == the locked frequency. If not, it is applied
     through env/rootdo.sh (the owner allows jetson_clocks whenever needed) and re-checked
  4. GPU current frequency within 2 % of the locked frequency
  5. no other GPU job and no stale tegrastats: nvidia-smi utilization.gpu is 0 % in every sample of a short window
     (on Thor nvidia-smi lists no processes, but its utilisation is system-wide: it read 98 % under a GEMM loop and
     0 % at idle, T002), and `pgrep tegrastats` is empty
Records: everything above plus EMC rate, CPU frequencies, temperatures, soctherm over-current counters, fan.

python bench/gate.py [--no-apply] [--idle-window-s 3]      # prints JSON; exit 1 if the gate fails
from gate import run_gate; state = run_gate()                 # raises GateFailed
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTDO = os.path.join(REPO, "env", "rootdo.sh")
DEVFREQ = "/sys/class/devfreq/gpu-gpc-0"
MAX_PERF_MODE = {"id": "0", "name": "MAXN"}      # env/README.md, T001
EXPECTED_SMS = 20                                  # env/README.md, T001
LOCKED_HZ_DEFAULT = 1575000000                     # devfreq max in MAXN; T002 measures the clock under load


class GateFailed(RuntimeError):
    pass


def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception as exc:
        return f"n/a: {exc}"


def sh(args, root=False, timeout=60):
    r = subprocess.run(([ROOTDO] if root else []) + args, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def gpu_util_samples(window_s=3.0, n=6):
    out = []
    for _ in range(n):
        v = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        out.append(int(v) if v.strip().isdigit() else v)
        time.sleep(window_s / n)
    return out


def locked_hz():
    """The locked frequency T002 measured under load, if hw_constants exists; else the devfreq max."""
    for p in sorted(glob.glob(os.path.join(REPO, "results", "*_E0_platform*", "hw_constants.json"))):
        try:
            v = json.load(open(p)).get("gpu_clock_locked_hz", {})
            v = v.get("value") if isinstance(v, dict) else v
            if isinstance(v, (int, float)):
                return int(v), os.path.relpath(p, REPO)
        except Exception:
            pass
    return LOCKED_HZ_DEFAULT, "default: devfreq max_freq in MAXN"


def device_record():
    temps = {rd(f"{z}/type"): rd(f"{z}/temp") for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*"))}
    oc = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(f"{h}/name") == "soctherm_oc":
            for f in sorted(glob.glob(f"{h}/oc*_event_cnt")):
                oc[os.path.basename(f)] = rd(f)
    fan = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(f"{h}/name") == "pwmfan":
            fan = {"pwm1": rd(f"{h}/pwm1"), "pwm1_enable": rd(f"{h}/pwm1_enable")}
    cpu = sorted({rd(p) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")})
    return {"thermal_zones_mC": temps, "soctherm_oc_event_cnt": oc, "fan": fan, "cpu_cur_freq_khz_set": cpu,
            "cpu_online": rd("/sys/devices/system/cpu/online"),
            "emc_hz": sh(["cat", "/sys/kernel/debug/bpmp/debug/clk/emc/rate"], root=True)}


def run_gate(apply_clocks=True, idle_window_s=3.0, check_idle=True, sm_count=None):
    st = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    q = sh(["/usr/sbin/nvpmodel", "-q"])
    m = re.search(r"NV Power Mode:\s*(\S+)\s*\n\s*(\d+)", q)
    mode = {"id": m.group(2), "name": m.group(1)} if m else {"id": "n/a", "name": q}
    st["power_mode"] = mode
    st["checks"]["1_power_mode"] = mode == MAX_PERF_MODE
    if sm_count is None:
        import torch
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    st["sm_count"] = sm_count
    st["checks"]["2_sm_count"] = sm_count == EXPECTED_SMS
    lk, lk_src = locked_hz()
    st["gpu_clock_locked_hz"], st["gpu_clock_locked_source"] = lk, lk_src
    mn, mx = int(rd(f"{DEVFREQ}/min_freq")), int(rd(f"{DEVFREQ}/max_freq"))
    if apply_clocks and not (mn == mx == lk):
        st["jetson_clocks_applied_by_gate"] = sh(["jetson_clocks"], root=True) or "ok"
        sh(["jetson_clocks", "--fan"], root=True)
        mn, mx = int(rd(f"{DEVFREQ}/min_freq")), int(rd(f"{DEVFREQ}/max_freq"))
    st["gpu_min_hz"], st["gpu_max_hz"] = mn, mx
    st["jetson_clocks"] = mn == mx == lk
    st["checks"]["3_jetson_clocks"] = st["jetson_clocks"]
    cur = int(rd(f"{DEVFREQ}/cur_freq"))
    st["gpu_hz"] = cur
    st["checks"]["4_gpu_freq_within_2pct"] = abs(cur - lk) <= 0.02 * lk
    teg = subprocess.run(["pgrep", "-a", "tegrastats"], capture_output=True, text=True).stdout.strip()
    st["stale_tegrastats"] = teg or "none"
    util = gpu_util_samples(idle_window_s) if check_idle else "n/a: not checked"
    st["gpu_util_pct_samples"] = util
    st["checks"]["5_no_other_gpu_job"] = (not teg) and (not check_idle or all(u == 0 for u in util))
    st.update(device_record())
    st["jetson_clocks_show"] = sh(["jetson_clocks", "--show"], root=True)
    st["gate_pass"] = all(st["checks"].values())
    return st


def wait_for_gate(max_wait_s=3600, poll_s=60, **kw):
    """Retry while only check 5 (another GPU job) fails; any other failure raises at once."""
    t0 = time.time()
    while True:
        st = run_gate(**kw)
        if st["gate_pass"]:
            return st
        others = [k for k, v in st["checks"].items() if not v and k != "5_no_other_gpu_job"]
        if others or time.time() - t0 > max_wait_s:
            raise GateFailed(json.dumps({k: st[k] for k in ("checks", "power_mode", "gpu_hz", "gpu_util_pct_samples")}))
        print(f"gate: GPU busy {st['gpu_util_pct_samples']}; waiting {poll_s} s", file=sys.stderr, flush=True)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-apply", action="store_true")
    ap.add_argument("--idle-window-s", type=float, default=3.0)
    a = ap.parse_args()
    st = run_gate(apply_clocks=not a.no_apply, idle_window_s=a.idle_window_s)
    print(json.dumps(st, indent=1))
    sys.exit(0 if st["gate_pass"] else 1)


if __name__ == "__main__":
    main()
