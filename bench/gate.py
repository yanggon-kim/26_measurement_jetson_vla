"""Protocol §1 device gate for this Thor, and the device-state record every run needs.

Checks (all must pass for a measured or profiled run):
  1. power mode is the maximum-performance mode recorded in env/README.md (MAXN, id 0)
  2. SM count equals the count T001 recorded in that mode (20)
  3. jetson_clocks applied in this boot: GPU devfreq min == max == the locked frequency. If not, it is applied
     through env/rootdo.sh (the owner allows jetson_clocks whenever needed) and re-checked
  4. GPU current frequency within 2 % of the locked frequency
  5. no other GPU job and no stale tegrastats: nvidia-smi utilization.gpu ≤ 2 % in every sample of a 5 s window
     (on Thor nvidia-smi lists no processes, but its utilisation is system-wide: it read 98 % under a GEMM loop and
     0 % at idle, T002), and `pgrep tegrastats` is empty. A job that starts later is not caught here: harnesses take
     oc_counts() before and after and, where they time, keep the per-run utilisation / OC record
  6. the soctherm OC counters are readable (the throttle verdict depends on them)
  Check 3 covers the whole jetson_clocks state: GPU min == max == locked, EMC override at max, CPUs min == max, fan 255.
Records: everything above plus EMC rate, CPU frequencies, temperatures, soctherm over-current counters, fan.

python bench/gate.py [--no-apply] [--idle-window-s 5]      # prints JSON; exit 1 if the gate fails
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
IDLE_UTIL_MAX_PCT = 2   # an idle GPU reads 0 % with stray 1 % samples (T002: 2 of 62); a GEMM loop reads 98 %


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


def rd_int(p):
    """int of a sysfs node, or None if it cannot be read (the gate then fails that check instead of crashing)."""
    try:
        return int(rd(p))
    except (TypeError, ValueError):
        return None


def gpu_util_samples(window_s=5.0, n=10):
    out = []
    for _ in range(n):
        v = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        out.append(int(v) if v.strip().isdigit() else v)
        time.sleep(window_s / n)
    return out


def locked_hz():
    """The locked frequency of the NEWEST max-performance-mode platform record that has one; else the devfreq max.

    Only E0 directories whose hw_constants.json says power_mode = MAXN count; newest first (directory names start with
    the date), so a later authoritative re-run supersedes an earlier one."""
    for p in sorted(glob.glob(os.path.join(REPO, "results", "*_E0_platform*", "hw_constants.json")), reverse=True):
        try:
            hc = json.load(open(p))
            if hc.get("power_mode", {}).get("name") != MAX_PERF_MODE["name"]:
                continue
            v = hc.get("gpu_clock_locked_hz", {})
            v = v.get("value") if isinstance(v, dict) else v
            if isinstance(v, (int, float)):
                return int(v), os.path.relpath(p, REPO)
        except Exception:
            pass
    return LOCKED_HZ_DEFAULT, "default: devfreq max_freq in MAXN"


def device_record():
    temps = {rd(f"{z}/type"): rd(f"{z}/temp") for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*"))}
    fan = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(f"{h}/name") == "pwmfan":
            fan = {"pwm1": rd(f"{h}/pwm1"), "pwm1_enable": rd(f"{h}/pwm1_enable")}
    cpu = sorted({rd(p) for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")})
    return {"thermal_zones_mC": temps, "soctherm_oc_event_cnt": oc_counts(), "fan": fan, "cpu_cur_freq_khz_set": cpu,
            "cpu_online": rd("/sys/devices/system/cpu/online"),
            "emc_hz": sh(["cat", "/sys/kernel/debug/bpmp/debug/clk/emc/rate"], root=True)}


def oc_counts():
    """soctherm over-current event counters (system-wide). On this Thor VDD_GPU's ina3221 warning limit is 5,060 mA
    (≈ 100 W at 19.76 V); sustained heavy GPU load reaches it and oc3 (throttle enabled) fires while devfreq still
    reports the locked clock (T002). Take this before and after every measured run; any delta marks the run throttled.
    Because the counters are system-wide, another user's load also counts: conservative."""
    out = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if rd(f"{h}/name") == "soctherm_oc":
            for f in sorted(glob.glob(f"{h}/oc*_event_cnt")):
                out[os.path.basename(f).replace("_event_cnt", "")] = rd_int(f)
    return out


def throttle_verdict(before, after, gpu_hz_samples=None, locked=None):
    """throttled = any OC event during the run, or a devfreq sample below the locked clock.
    If the OC counters cannot be read on either side, the verdict is "unknown", never "not throttled"."""
    if locked is None:
        locked = locked_hz()[0]
    keys = sorted(set(before) | set(after))
    readable = bool(keys) and all(isinstance(before.get(k), int) and isinstance(after.get(k), int) for k in keys)
    delta = {k: (after[k] - before[k]) if readable else "n/a" for k in keys}
    low = [f for f in (gpu_hz_samples or []) if f < locked]
    if not readable:
        return {"throttled": "unknown: soctherm OC counters not readable", "oc_event_delta": delta,
                "devfreq_samples_below_locked": len(low)}
    return {"throttled": any(v > 0 for v in delta.values()) or bool(low), "oc_event_delta": delta,
            "devfreq_samples_below_locked": len(low)}


def clocks_locked(lk):
    """jetson_clocks state: GPU devfreq min == max == locked, EMC override on at max, CPUs min == max, fan at 255."""
    show = sh(["jetson_clocks", "--show"], root=True)
    mn, mx = rd_int(f"{DEVFREQ}/min_freq"), rd_int(f"{DEVFREQ}/max_freq")
    emc = re.search(r"EMC MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+) FreqOverride=(\d)", show)
    cpus = re.findall(r"cpu\d+:.*?MinFreq=(\d+) MaxFreq=(\d+)", show)
    fan = re.search(r"hwmon\d+_pwm1=(\d+)", show)
    parts = {"gpu_min_eq_max_eq_locked": mn == mx == lk,
             "emc_override_at_max": bool(emc) and emc.group(4) == "1" and emc.group(3) == emc.group(2),
             "cpu_min_eq_max": bool(cpus) and all(a == b for a, b in cpus),
             "fan_pwm_255": bool(fan) and fan.group(1) == "255"}
    return parts, show, mn, mx


def run_gate(apply_clocks=True, idle_window_s=5.0, check_idle=True, sm_count=None):
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
    parts, show, mn, mx = clocks_locked(lk)
    if apply_clocks and not all(parts.values()):
        st["jetson_clocks_applied_by_gate"] = {"before": parts}
        sh(["jetson_clocks"], root=True)
        sh(["jetson_clocks", "--fan"], root=True)
        parts, show, mn, mx = clocks_locked(lk)
    st["gpu_min_hz"], st["gpu_max_hz"] = mn, mx
    st["jetson_clocks_parts"] = parts
    st["jetson_clocks"] = all(parts.values())
    st["checks"]["3_jetson_clocks"] = st["jetson_clocks"]
    cur = rd_int(f"{DEVFREQ}/cur_freq")
    st["gpu_hz"] = cur if cur is not None else "n/a: unreadable"
    st["checks"]["4_gpu_freq_within_2pct"] = cur is not None and abs(cur - lk) <= 0.02 * lk
    teg = subprocess.run(["pgrep", "-a", "tegrastats"], capture_output=True, text=True).stdout.strip()
    st["stale_tegrastats"] = teg or "none"
    util = gpu_util_samples(idle_window_s) if check_idle else "n/a: not checked"
    st["gpu_util_pct_samples"] = util
    st["checks"]["5_no_other_gpu_job"] = (not teg) and (not check_idle or all(
        isinstance(u, int) and u <= IDLE_UTIL_MAX_PCT for u in util))
    oc = oc_counts()
    st["checks"]["6_oc_counters_readable"] = bool(oc) and all(isinstance(v, int) for v in oc.values())
    st.update(device_record())
    st["jetson_clocks_show"] = show
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
    ap.add_argument("--idle-window-s", type=float, default=5.0)
    a = ap.parse_args()
    st = run_gate(apply_clocks=not a.no_apply, idle_window_s=a.idle_window_s)
    print(json.dumps(st, indent=1))
    sys.exit(0 if st["gate_pass"] else 1)


if __name__ == "__main__":
    main()
