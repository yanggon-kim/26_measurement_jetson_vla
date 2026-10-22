"""T004 follow-up: does a model's latency follow the hidden memory-path state (results/20261022_E1_state_probe/)?

Loads one model through bench/run_e3.py's adapter, instruments the regions, then for --iters iterations:
memory_state_probe() (≈ 3 ms of GPU) -> one inference -> region times. Writes per iteration: probe µs, state, e2e and
region ms, GPU temperature, VDD_GPU power. Run once per precision setting.

python bench/state_vs_latency.py --model groot_n17 --precision eager_fp32 --iters 150 --out results/<run>/<name>.csv
"""
import argparse
import csv
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate as G  # noqa: E402
import regions as R  # noqa: E402
import run_e3 as E3  # noqa: E402


def vdd_gpu_w():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if open(f"{h}/name").read().strip() == "ina3221":
            for n in "123":
                if open(f"{h}/in{n}_label").read().strip() == "VDD_GPU":
                    return int(open(f"{h}/in{n}_input").read()) * int(open(f"{h}/curr{n}_input").read()) / 1e6
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", choices=["eager_fp32", "shipped"], required=True)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g0 = G.wait_for_gate()
    E3.set_precision(a.precision)
    ad = E3.make_adapter(a.model)
    ad.set_n(a.n or ad.n_default)
    rec = R.Recorder()
    undo = ad.instrument(rec)
    for _ in range(20):
        rec.reset()
        ad.iteration()
    rows = []
    for it in range(a.iters):
        ms = G.memory_state_probe()
        rec.reset()
        rec.start()
        walls, _ = ad.iteration()
        rec.stop()
        r = rec.results_ms(ad.derive_ae_total)
        rows.append({"iter": it, "probe_us": ms["probe_us_per_call"], "state": ms["state"], "e2e_ms": walls["e2e_wall"],
                     "vision_ms": r.get("vision"), "vlm_prefix_ms": r.get("vlm_prefix"), "ae_total_ms": r.get("ae_total"),
                     "gpu_C": int(G.rd("/sys/class/thermal/thermal_zone1/temp") or 0) / 1e3, "vdd_gpu_W": round(vdd_gpu_w(), 2),
                     "t_s": round(time.time(), 3)})
    undo()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fast = [r for r in rows if r["state"] == "fast"]
    norm = [r for r in rows if r["state"] == "normal"]
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    print(f"{a.model} {a.precision}: fast {len(fast)} it, ae {med([r['ae_total_ms'] for r in fast]):.1f} ms; normal {len(norm)} it, "
          f"ae {med([r['ae_total_ms'] for r in norm]):.1f} ms; gate {g0['checks']}")


if __name__ == "__main__":
    main()
