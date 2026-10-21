"""T005 — do the four INA3221 rails double-count? (protocol §4)

E7 will report energy per inference by integrating the sysfs power rails. That only works if
we know what the rails mean: whether they are four independent branches that may be summed, or
whether one of them already contains the others, in which case summing inflates every joule
the campaign reports.

This device exposes four:

    VDD_GPU_SOC        measured on the ~19.9 V input side
    VDD_CPU_CV         measured on the ~19.9 V input side
    VIN_SYS_5V0        the 5 V system rail
    VDDQ_VDD2_1V8AO    memory

The test is behavioural: drive a load that is almost purely GPU-compute, and one that is
almost purely DRAM traffic, and see which rails respond and by how much. A rail that contains
another must rise whenever that other one rises.

    python bench/power_rails_check.py --outdir results/<date>_E7_rails --seconds 60
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import (REPO, collect_raw, ina3221_rails, sh_out,  # noqa: E402
                    software_versions, write_json)


class Sampler(threading.Thread):
    """Poll every rail at ~10 Hz on a separate thread while the GPU is busy."""

    def __init__(self, hz: float = 10.0):
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        self.stop = threading.Event()
        self.samples: list[dict] = []

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            r = ina3221_rails()
            self.samples.append({"t": t, **{k: v.get("mW") for k, v in r.items()},
                                 **{f"{k}__mA": v.get("mA") for k, v in r.items()},
                                 **{f"{k}__mV": v.get("mV") for k, v in r.items()}})
            time.sleep(max(0.0, self.period - (time.time() - t)))


def measure(load_fn, seconds: float, tag: str, settle: float = 3.0) -> tuple[dict, list[dict]]:
    """Run the load for `settle` seconds BEFORE sampling starts, so the rails are at their
    steady value. Without it each mean carries a second or two of ramp, which showed up as a
    `min_mW` equal to the idle reading."""
    t_settle = time.time() + settle
    while time.time() < t_settle:
        load_fn()
    torch.cuda.synchronize()
    s = Sampler()
    s.start()
    t_end = time.time() + seconds
    while time.time() < t_end:
        load_fn()
    torch.cuda.synchronize()
    s.stop.set()
    s.join(timeout=2)
    rails = [k for k in s.samples[0] if k != "t" and "__" not in k]
    out = {}
    for r in rails:
        v = [x[r] for x in s.samples if x.get(r) is not None]
        out[r] = {"mean_mW": round(statistics.mean(v), 1),
                  "median_mW": round(statistics.median(v), 1),
                  "min_mW": min(v), "max_mW": max(v), "n": len(v)}
    print(f"  {tag:22s} " + "  ".join(f"{r}={out[r]['mean_mW']:6.0f}mW" for r in rails))
    return out, s.samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate))
    print(f"\nsampling four rails at 10 Hz, {args.seconds:.0f} s per load\n")

    # --- loads
    n = 256 * 1024 * 1024 // 2
    xc = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    zc = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    g = 8192
    ag = torch.empty(g, g, device="cuda", dtype=torch.bfloat16).normal_()
    bg = torch.empty(g, g, device="cuda", dtype=torch.bfloat16).normal_()

    def idle():
        time.sleep(0.05)

    def clone_loop():
        for _ in range(4):
            xc.clone()

    def copy_kernel_loop():
        for _ in range(4):
            torch.mul(xc, 2.0, out=zc)

    def gemm_loop():
        for _ in range(2):
            ag @ bg

    loads = [
        ("idle", idle, "GPU idle; the baseline every other row is measured against"),
        ("clone_256mib", clone_loop,
         "clone() of a 256 MiB tensor in a loop — DRAM traffic through the copy engine"),
        ("copy_kernel_256mib", copy_kernel_loop,
         "z = 2*x over 256 MiB in a loop — the same DRAM traffic as a kernel"),
        ("gemm_8192_bf16", gemm_loop,
         "8192^2 bf16 GEMM in a loop — almost pure tensor-core compute"),
    ]

    results, raw = {}, {}
    for tag, fn, _ in loads:
        results[tag], raw[tag] = measure(fn, args.seconds, tag)

    rails = sorted(results["idle"])
    rows = []
    for tag, _, desc in loads:
        for r in rails:
            rows.append({
                "load": tag, "rail": r,
                "mean_mW": results[tag][r]["mean_mW"],
                "median_mW": results[tag][r]["median_mW"],
                "min_mW": results[tag][r]["min_mW"], "max_mW": results[tag][r]["max_mW"],
                "n": results[tag][r]["n"],
                "above_idle_mW": round(results[tag][r]["mean_mW"]
                                       - results["idle"][r]["mean_mW"], 1),
                "above_idle_median_mW": round(results[tag][r]["median_mW"]
                                              - results["idle"][r]["median_mW"], 1),
                "description": desc,
            })
        rows.append({
            "load": tag, "rail": "SUM_OF_FOUR",
            "mean_mW": round(sum(results[tag][r]["mean_mW"] for r in rails), 1),
            "median_mW": "", "min_mW": "", "max_mW": "", "n": "",
            "above_idle_mW": round(sum(results[tag][r]["mean_mW"] - results["idle"][r]["mean_mW"]
                                       for r in rails), 1),
            "above_idle_median_mW": round(sum(results[tag][r]["median_mW"]
                                              - results["idle"][r]["median_mW"]
                                              for r in rails), 1),
            "description": "arithmetic sum of the four rails — valid only if they are "
                           "independent, which is what this run tests",
        })

    with (out / "power_rails_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for tag in raw:
        with (out / "raw" / f"rail_samples_{tag}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw[tag][0].keys()))
            w.writeheader()
            w.writerows(raw[tag])

    print("\n=== above idle, mW ===")
    print(f"{'load':22s} " + " ".join(f"{r:>18s}" for r in rails) + f" {'SUM':>10s}")
    for tag, _, _ in loads:
        if tag == "idle":
            continue
        d = {r: results[tag][r]["mean_mW"] - results["idle"][r]["mean_mW"] for r in rails}
        print(f"{tag:22s} " + " ".join(f"{d[r]:18.0f}" for r in rails)
              + f" {sum(d.values()):10.0f}")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E7-prep (rail double-count check)", "task": "T005",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": None, "precision": "bfloat16", "compile_mode": "none (eager)",
        "model_repo": None, "model_commit": None, "checkpoint": None,
        "num_denoise_steps": None, "action_horizon": None,
        "prefix_tokens": None, "suffix_tokens": None,
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "batch": 1, "n_warmup": "3 s settle per load", "n_timed": "60 s per load at 10 Hz",
        "seconds_per_load": args.seconds, "sample_hz": 10, "settle_seconds": 3.0,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out(
            "sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/power_rails_check.py --outdir {args.outdir} "
                   f"--seconds {args.seconds}",
        "rails": rails, "results": results,
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "raw_files": collect_raw(out, "bench/power_rails_check.py"),
        "notes": [
            "Rails are read from /sys/bus/i2c/drivers/ina3221/*/hwmon/*, mW = mV x mA / 1000.",
            "The 'sum of shunt voltages' pseudo-channel is skipped.",
            "Sampling runs on a separate thread at 10 Hz while the GPU is loaded; the load "
            "loop and the sampler are independent, so the sample count per load varies.",
        ],
    })
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
