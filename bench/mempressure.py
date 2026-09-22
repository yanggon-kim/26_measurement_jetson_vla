"""T010 / E11e — how much the GPU prefix slows down when memory bandwidth is taken away.

The hybrid model assumes the GPU-side prefix runs at full speed while the memory system is busy
denoising: `prefix_slowdown = 1.0`, marked **[ASSUMPTION]** in `docs/00_context.md`. There is no
PIM to test that against, so protocol §13.5 specifies a software proxy — and is explicit that
it is a proxy:

  "A real PIM occupies DRAM channels through the command bus and blocks ordinary accesses to
   them; a software hog instead competes at the memory controller and in the caches, and
   additionally takes GPU time slices (`gpu_hog`) or CPU cores (`cpu_hog`). The controls bound
   those side effects; the result is the sensitivity of the GPU work to lost memory bandwidth,
   **not a measurement of PIM interference**."

The two spin controls are what make the result interpretable: `gpu_spin_control` runs the same
duty cycle with cache-resident GEMMs (GPU time taken, no bandwidth taken) and
`cpu_spin_control` the same worker count on the same cores doing in-cache arithmetic. Whatever
slowdown they cause is *not* lost bandwidth.

    python bench/mempressure.py --outdir-prefix results/<date>_E11e_pi0 --seconds 60
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import phase_loops as PL  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, ina3221_rails,  # noqa: E402
                    sh_out, software_versions, write_json)
from host_segments import ModelInputs  # noqa: E402

PEAK_GBPS = 187.47              # T005/E1 measured, results/20260922_E1_peak/hw_constants.json
FG_CORES = "0-5"
BG_CORES = "6,7,8,9,10,11"


def start_hog(kind, duty, seconds, log):
    cmd = [sys.executable, str(REPO / "bench" / "bw_hog.py"), "--kind", kind,
           "--duty", str(duty), "--seconds", str(seconds + 20), "--log", str(log)]
    if kind.startswith("cpu"):
        cmd += ["--cores", BG_CORES, "--workers", "6"]
        cmd = ["taskset", "-c", BG_CORES] + cmd
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def hog_rate(log: Path) -> float | None:
    if not log.exists():
        return None
    vals = []
    for ln in log.read_text().splitlines():
        try:
            vals.append(json.loads(ln)["rate"])
        except Exception:                                          # noqa: BLE001
            pass
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def start_tegrastats(log: Path):
    if log.exists():
        log.unlink()                       # tegrastats --logfile APPENDS (T001 finding)
    return subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "100",
                             "--logfile", str(log)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir-prefix", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--modes", default="eager,reduce-overhead")
    ap.add_argument("--phases", default="prefix")
    args = ap.parse_args()
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5})
    except OSError as e:                                           # noqa: BLE001
        print(f"  could not pin foreground: {e}")

    loads = ([("none", None, 0)]
             + [("gpu_hog", "gpu", d) for d in (25, 50, 75, 100)]
             + [("cpu_hog", "cpu", d) for d in (25, 50, 75, 100)]
             + [("gpu_spin_control", "gpu_spin_control", 100),
                ("cpu_spin_control", "cpu_spin_control", 100)])

    for mode in [m for m in args.modes.split(",") if m]:
        out = Path(f"{args.outdir_prefix}_{mode}_mempressure")
        (out / "raw").mkdir(parents=True, exist_ok=True)
        cm = None if mode == "eager" else mode
        policy, cfg, shipped = P.load_policy(num_inference_steps=args.steps, compile_mode=cm)
        real = P.assert_real_weights(policy)
        fx = P.load_fixture()
        batch, _, _ = P.build_batch(policy, fx)
        print(f"\n=== {mode} ===", flush=True)
        rows, base = [], {}
        for phase in [p for p in args.phases.split(",") if p]:
            fn = PL.make_phase(policy, batch, phase, args.steps)
            for label, kind, duty in loads:
                tag = label if duty == 0 or label.endswith("control") else f"{label}_{duty}"
                hog_log = out / "raw" / f"bw_hog_{tag}.log"
                teg_log = out / "raw" / f"tegrastats_{phase}_{tag}.log"
                alone = None
                proc = None
                if kind:
                    # the background rate ALONE, before the foreground starts (§13.5)
                    a = start_hog(kind, duty, 8, out / "raw" / f"bw_hog_{tag}_alone.log")
                    a.wait(timeout=60)
                    alone = hog_rate(out / "raw" / f"bw_hog_{tag}_alone.log")
                    proc = start_hog(kind, duty, args.seconds, hog_log)
                    time.sleep(3)
                teg = start_tegrastats(teg_log)
                r = PL.run_loop(fn, args.seconds, f"{phase}/{tag}")
                teg.terminate()
                during = hog_rate(hog_log)
                if proc:
                    proc.terminate()
                    proc.wait(timeout=30)
                samples = r.pop("samples")
                if label == "none":
                    base[phase] = r["median_ms"]
                slow = r["median_ms"] / base[phase] if base.get(phase) else ""
                rows.append({"load_label": tag, "duty_pct": duty, "phase": phase,
                             "bg_GBps_alone": round(alone, 3) if alone else "",
                             "bg_GBps_during": round(during, 3) if during else "",
                             "pct_of_measured_peak": round(100 * during / PEAK_GBPS, 1)
                             if during and kind in ("gpu", "cpu") else "",
                             "region": phase, "median_ms": r["median_ms"],
                             "p5_ms": r["p5_ms"], "p95_ms": r["p95_ms"], "n": r["n"],
                             "throughput_inf_per_s": r["throughput_inf_per_s"],
                             "slowdown": round(slow, 4) if slow else ""})
                with (out / "raw" / f"latency_samples_{phase}_{tag}.csv").open(
                        "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iter", "ms"])
                    w.writerows(enumerate(samples))
                print(f"  {phase}/{tag:22s} {r['median_ms']:8.2f} ms  "
                      f"{r['throughput_inf_per_s']:6.2f} inf/s  slowdown "
                      f"{rows[-1]['slowdown']}  bg {rows[-1]['bg_GBps_during']} GB/s",
                      flush=True)

        with (out / "mempressure.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # --- CPU utilisation per phase, in a SEPARATE run (§13.6)
        cpu_rows = []
        for phase in ("prefix", "ae", "full"):
            fn = PL.make_phase(policy, batch, phase, args.steps)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0, c0, p0 = time.perf_counter(), time.thread_time(), time.process_time()
            n = 0
            while time.perf_counter() - t0 < 30.0:
                fn()
                n += 1
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            cpu_rows.append({"phase": phase, "scope": "thread",
                             "pct": round(100 * (time.thread_time() - c0) / wall, 2), "n": n})
            cpu_rows.append({"phase": phase, "scope": "process",
                             "pct": round(100 * (time.process_time() - p0) / wall, 2), "n": n})
            print(f"  cpu_util {phase:8s} thread {cpu_rows[-2]['pct']:6.2f} %  "
                  f"process {cpu_rows[-1]['pct']:6.2f} %", flush=True)
        with (out / "cpu_util.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "scope", "pct", "n"])
            w.writeheader()
            w.writerows(cpu_rows)

        # --- power per phase, from the four rails
        prows = []
        for phase in ("prefix", "ae", "full"):
            fn = PL.make_phase(policy, batch, phase, args.steps)
            acc = {}
            t_end = time.perf_counter() + 20
            while time.perf_counter() < t_end:
                fn()
                for rail, d in ina3221_rails().items():
                    acc.setdefault(rail, []).append(d.get("mW") or 0)
            for rail, vals in acc.items():
                prows.append({"phase": phase, "rail": rail,
                              "mean_mW": round(sum(vals) / len(vals), 1), "n": len(vals)})
            prows.append({"phase": phase, "rail": "module",
                          "mean_mW": round(sum(sum(v) / len(v) for v in acc.values()), 1),
                          "n": len(next(iter(acc.values())))})
        with (out / "power.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "rail", "mean_mW", "n"])
            w.writeheader()
            w.writerows(prows)

        mi = ModelInputs("pi0", P.REVISION, sh_out(f"git -C {REPO} rev-parse HEAD"),
                         "MAXN+jetson_clocks", 816, 51, args.steps)
        style = "eager" if cm is None else "compiled"
        for r in rows:
            if r["slowdown"] != "":
                mi.add("prefix_slowdown", r["slowdown"], "ratio", "value", r["n"],
                       "mempressure.csv", key=r["load_label"], exec_style=style,
                       timing="wall_loop")
            for which, col in (("alone", "bg_GBps_alone"), ("during", "bg_GBps_during")):
                if r[col] != "":
                    mi.add("bg_bw_GBps", r[col], "GBps", "median", 1, "mempressure.csv",
                           key=r["load_label"], variant=f"when={which}")
            mi.add("prefix_throughput", r["throughput_inf_per_s"], "inf_per_s", "value",
                   r["n"], "mempressure.csv", variant=f"load={r['load_label']}",
                   exec_style=style, timing="wall_loop")
        for c in cpu_rows:
            mi.add("cpu_util", c["pct"], "pct", "value", c["n"], "cpu_util.csv",
                   key={"prefix": "vlm_prefix", "ae": "ae_total", "full": "e2e"}[c["phase"]],
                   variant=f"scope={c['scope']}", exec_style=style)
        for pr in prows:
            mi.add("p_mean_W", pr["mean_mW"] / 1000.0, "W", "mean", pr["n"], "power.csv",
                   key=f"{pr['phase']}/{pr['rail']}", exec_style=style)
        n_mi = mi.write(out / "model_inputs.csv")

        for f_ in sorted((out / "raw").glob("latency_samples_*.csv")):
            compress_and_record(f_)
        sw = software_versions()
        write_json(out / "meta.json", {
            "experiment": "E11e", "task": "T010",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": P.CHECKPOINT, "model_repo": P.CHECKPOINT, "model_commit": P.REVISION,
            "model_revision": P.REVISION, "checkpoint": P.CHECKPOINT, "precision": "bfloat16",
            "compile_mode": cm, "batch": 1, "num_denoise_steps": args.steps,
            "action_horizon": getattr(cfg, "chunk_size", None),
            "prefix_tokens": 816, "suffix_tokens": 51,
            "n_warmup": 5, "n_timed": f"{args.seconds} s loop per condition",
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
            "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
            "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
            "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
            "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
            "python": sw.get("python"), "venv": sw.get("executable"),
            "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
            "bench_files_sha256": sw.get("bench_files_sha256"),
            "weights_verified": real, "shipped_config": shipped,
            "command": f"python bench/mempressure.py --outdir-prefix {args.outdir_prefix}",
            "measured_peak_GBps": PEAK_GBPS,
            "foreground_cores": FG_CORES, "background_cores": BG_CORES,
            "hog_period_ms": 20, "hog_buffer_MiB": 256,
            "throttled": False,
            "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
            "model_inputs_rows": n_mi, "raw_files": [],
            "notes": [
                "PROXY, NOT PIM. Protocol §13.5: a software hog competes at the memory "
                "controller and in the caches and additionally takes GPU time slices or CPU "
                "cores; the two spin controls bound those side effects. The result is the "
                "sensitivity of the GPU work to lost bandwidth, not PIM interference.",
                "Background GB/s counts read + write, the same convention as E1's clone() "
                "row, and is reported alone and during the foreground run.",
                "Foreground pinned to cores 0-5, the CPU hog to 6-11; both recorded.",
                "tegrastats --logfile appends, so each log is deleted before its run (T001).",
                "E1 measured bandwidth varying run to run by up to 1.19x with clocks and "
                "temperature constant, which is the noise floor under every slowdown here.",
            ],
        })
        meta = json.loads((out / "meta.json").read_text())
        meta["raw_files"] = collect_raw(out, meta["command"])
        write_json(out / "meta.json", meta)
        print(f"wrote {out} ({n_mi} rows)", flush=True)
        del policy
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
