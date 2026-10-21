"""The E7 power-measurement engine, shared by the LeRobot harness and GR00T.

GR00T lives in `~/venvs/groot` (torch 2.10) and the LeRobot harness in `~/venvs/pi0`
(torch 2.8), so `bench/power_phases.py` — which imports lerobot through `pi0_common` — is
not importable from the GR00T side. The alternative to this module is a second
implementation of the rail sampler, the tegrastats handling and the row schema, which is
how two sets of power numbers end up not comparable.

**This module imports nothing but the standard library and `bench/common.py`** (itself
stdlib-only at module scope), exactly like `bench/graph_clone.py`. `run_loop` is passed in
rather than imported, because that is the only piece that differs between the two venvs.

The protocol §4 rules live here so that both callers obey them without either being trusted
to remember:

  * all four INA3221 rails, sampled at 20 Hz, each reported separately **and** as `module`
    (all four summed) and `module_without_memory_rail` (three);
  * the memory rail is emitted as **`VDDQ_VDD2_1V8AO (regulator input)`**, measured at the
    regulator input (~5.04 V) and never rescaled to 1.8 V. It is an upper bound on DRAM
    power and its above-idle delta sits next to the absolute value in every row;
  * an idle baseline in the same power mode precedes the loops, and every row carries both
    the absolute mean and the above-idle delta;
  * tegrastats runs alongside at `--interval 100`, its log deleted first because
    `--logfile` APPENDS (T001) and its process stopped through `tegrastats --stop` because
    terminating the `sudo` wrapper leaves the root process running (T010 left 22 alive,
    which corrupted the logs and added background load to every later condition).
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import threading
import time
from pathlib import Path

from common import ina3221_rails

MEM_RAIL = "VDDQ_VDD2_1V8AO"


class Rails(threading.Thread):
    """Samples every INA3221 rail from sysfs at `hz`, in the background."""

    def __init__(self, hz=20.0):
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        self.stop = threading.Event()
        self.samples: list[dict] = []

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            self.samples.append({"t": round(t, 4),
                                 **{k: v.get("mW") for k, v in ina3221_rails().items()}})
            time.sleep(max(0.0, self.period - (time.time() - t)))


def stop_tegrastats(proc=None):
    """Stop tegrastats for real.

    `Popen(["sudo", "tegrastats", ...]).terminate()` signals the *sudo* wrapper; the root
    tegrastats survives it and keeps appending. `tegrastats --stop` is the documented way;
    the pkill covers instances this process does not own.
    """
    subprocess.run(["sudo", "-n", "tegrastats", "--stop"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    subprocess.run(["sudo", "-n", "pkill", "-9", "-x", "tegrastats"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:                                          # noqa: BLE001
            pass


def measure(fn, seconds, tag, teg_log: Path, run_loop, hz: float = 20.0):
    """Run `fn` back-to-back for `seconds` while sampling the rails.

    `fn=None` measures the idle baseline: the rails are sampled for the same duration with
    nothing driving the GPU. `run_loop` is injected (bench/phase_loops.run_loop for the
    LeRobot models, groot_phases' own for GR00T) so this module imports neither.
    """
    # T016 step 4 (EV-10, 2026-09-26): warm up BEFORE the samplers start. The sampler used to
    # start first, so the power mean included the warm-up -- and a first call that compiles
    # (pi0 bf16_both prefix: ~45 s of max-autotune at idle-level power) -- which is why E7's
    # prefix + ae energies summed to 30 % less than `full`. Trimmed to the timed window the
    # residual is 0.7 % (results/..._E7r_pi0_energy_reconcile_MAXN).
    if fn is not None:
        import torch                                                  # noqa: PLC0415
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    stop_tegrastats()
    teg_log.parent.mkdir(parents=True, exist_ok=True)
    if teg_log.exists():
        teg_log.unlink()                       # --logfile APPENDS (T001)
    teg = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "100",
                            "--logfile", str(teg_log)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    s = Rails(hz)
    s.start()
    if fn is None:
        r = {"iterations": 0, "wall_s": seconds, "throughput_inf_per_s": 0.0,
             "median_ms": 0.0, "p5_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
        time.sleep(seconds)
    else:
        r = run_loop(fn, seconds, tag)
    s.stop.set()
    s.join(timeout=2)
    stop_tegrastats(teg)
    if not s.samples:
        raise SystemExit(f"no rail samples for {tag!r}: is /sys/bus/i2c/drivers/ina3221 "
                         f"readable? Refusing to write a power row with no power in it.")
    rails = [k for k in s.samples[0] if k != "t"]
    means = {k: statistics.fmean([x[k] for x in s.samples if x.get(k) is not None])
             for k in rails}
    r.pop("samples", None)
    return means, r, s.samples


def phase_rows(*, means, idle_means, r, n_samples, phase, steps, base_fields):
    """One row per rail for one phase, plus the two summed pseudo-rails.

    `base_fields` is whatever identifies the run (model, mode, exec_style, power_mode);
    it is prepended to every row so the CSV is self-describing.
    """
    module = sum(means.values())
    module_idle = sum(idle_means.values())
    no_mem = sum(v for k, v in means.items() if k != MEM_RAIL)
    no_mem_idle = sum(v for k, v in idle_means.items() if k != MEM_RAIL)
    lat_s = r["median_ms"] / 1e3
    rows = []
    for rail in sorted(idle_means) + ["module", "module_without_memory_rail"]:
        mw = {"module": module,
              "module_without_memory_rail": no_mem}.get(rail, means.get(rail))
        base = {"module": module_idle,
                "module_without_memory_rail": no_mem_idle}.get(rail, idle_means.get(rail))
        name = f"{rail} (regulator input)" if rail == MEM_RAIL else rail
        per_step = phase in ("ae", "full") and steps
        rows.append({
            **base_fields, "phase": phase, "rail": name,
            "mean_mW": round(mw, 1), "idle_mW": round(base, 1),
            "above_idle_mW": round(mw - base, 1),
            "iterations": r["iterations"], "median_ms": r["median_ms"],
            "throughput_inf_per_s": r["throughput_inf_per_s"],
            "energy_per_inference_mJ": round(mw * lat_s, 3),
            "energy_per_inference_above_idle_mJ": round((mw - base) * lat_s, 3),
            "energy_per_step_mJ": round(mw * lat_s / steps, 3) if per_step else "",
            "energy_per_step_above_idle_mJ":
                round((mw - base) * lat_s / steps, 3) if per_step else "",
            "n_samples": n_samples})
    return rows


def write_power_csv(out: Path, rows: list[dict], all_samples: list[dict]):
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    with (out / "power.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = sorted({k for x in all_samples for k in x})
    with (out / "raw" / "rail_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: x.get(k, "") for k in keys} for x in all_samples])


PROTOCOL_NOTES = [
    "The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to "
    "1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the "
    "absolute value, and never call it 'DRAM power' without those words (protocol §4).",
    "Each rail is reported separately, as module power (all four) and as the three-rail "
    "sum without the memory rail.",
    "T005 checked the four rails for double-counting and found none: a load that raises "
    "VDD_GPU_SOC by 51.8 W raises VIN_SYS_5V0 by only 8.7 W. They may be summed.",
    "An idle baseline was measured in this power mode, for this duration, before the loops.",
    "tegrastats --logfile appends, so each log is deleted before its run (T001).",
]
