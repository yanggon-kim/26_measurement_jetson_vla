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
import model_registry as MR  # noqa: E402
import phase_builders as PB  # noqa: E402
import pi0_common as P  # noqa: E402
import phase_loops as PL  # noqa: E402
from e11_summaries import e11e  # noqa: E402
from common import (CPU_UTIL_FIELDS, REPO, CpuSampler, collect_raw,  # noqa: E402
                    compress_and_record, ina3221_rails, sh_out, software_versions,
                    summarize, write_json)
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
    # Keep the hog's stderr. It used to go to DEVNULL with its stdout, so when every
    # cpu_spin_control worker died with UnboundLocalError the only evidence was an empty
    # rate log and a blank cell in the result table.
    err = Path(str(log) + ".stderr")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err.open("w"))


def hog_self_test(out: Path) -> None:
    """Prove every hog kind produces samples before spending 47 minutes on the sweep."""
    log = out / "raw" / "bw_hog_self_test.log"
    r = subprocess.run([sys.executable, str(REPO / "bench" / "bw_hog.py"), "--self-test",
                        "--workers", "6", "--cores", BG_CORES],
                       capture_output=True, text=True, timeout=600)
    log.write_text(r.stdout + r.stderr)
    for ln in r.stdout.splitlines():
        if ln.startswith("  ") or "self-test" in ln:
            print(ln, flush=True)
    if r.returncode != 0:
        raise SystemExit(f"bw_hog self-test failed; see {log}")


def hog_rate(log: Path, unit: str = "GBps") -> float | None:
    """Median rate of the requested UNIT.

    bw_hog emits GB/s for the copy hogs and both TFLOP/s and GB/s for the spin control. An
    earlier version ignored `unit` and stored whichever value came last, which put the spin
    control's 5.5 TFLOP/s into a GB/s column and into model_inputs.csv as bg_bw_GBps.
    """
    if not log.exists():
        return None
    vals = []
    for ln in log.read_text().splitlines():
        try:
            d = json.loads(ln)
            if d.get("unit", "GBps") == unit:
                vals.append(d["rate"])
        except Exception:                                          # noqa: BLE001
            pass
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def hog_duty(log: Path) -> float | None:
    """Median ACHIEVED duty cycle, which is a measurement; the requested duty is an input.

    The unit of work cannot be made arbitrarily small, so a hog asked for 25 % overshoots.
    Every analysis of these rows has to use this column, not `duty_pct`.
    """
    if not log.exists():
        return None
    vals = [json.loads(ln).get("achieved_duty_pct")
            for ln in log.read_text().splitlines() if ln.strip().startswith("{")]
    vals = sorted(v for v in vals if v is not None)
    return vals[len(vals) // 2] if vals else None


def stop_tegrastats(proc=None):
    """Stop tegrastats for real.

    `subprocess.Popen(["sudo", "tegrastats", ...]).terminate()` signals the *sudo* wrapper; the
    root tegrastats survives it and keeps appending to its log. E11e's first run left 22 of
    them alive, which both corrupted the logs (39,845 lines for a 60 s capture) and added
    growing background load to every later condition. `tegrastats --stop` is the documented
    way; the pkill is a belt-and-braces for instances it does not own.
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


def start_tegrastats(log: Path):
    stop_tegrastats()                      # never leave one running from a previous condition
    if log.exists():
        log.unlink()                       # tegrastats --logfile APPENDS (T001 finding)
    return subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "100",
                             "--logfile", str(log)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def measure_loads(out: Path, make_fn, phases, seconds: float) -> list[dict]:
    """The load sweep for each phase: none, gpu_hog/cpu_hog at 25-100 %, both spin controls (§13.5).

    `make_fn(phase)` returns the loop callable. Extracted from main() on 2026-09-28 so the GR00T driver
    (bench/groot_e11.py) runs the same sweep; the LeRobot path is unchanged. Writes mempressure.csv.
    """
    loads = ([("none", None, 0)]
             + [("gpu_hog", "gpu", d) for d in (25, 50, 75, 100)]
             + [("cpu_hog", "cpu", d) for d in (25, 50, 75, 100)]
             + [("gpu_spin_control", "gpu_spin_control", 100),
                ("cpu_spin_control", "cpu_spin_control", 100)])
    rows, base = [], {}
    for phase in phases:
        fn = make_fn(phase)
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
                proc = start_hog(kind, duty, seconds, hog_log)
                time.sleep(3)
            teg = start_tegrastats(teg_log)
            r = PL.run_loop(fn, seconds, f"{phase}/{tag}")
            stop_tegrastats(teg)
            during = hog_rate(hog_log)
            if proc:
                proc.terminate()
                proc.wait(timeout=30)
            samples = r.pop("samples")
            if label == "none":
                base[phase] = r["median_ms"]
            slow = r["median_ms"] / base[phase] if base.get(phase) else ""
            rows.append({"load_label": tag, "duty_pct": duty,
                         "duty_pct_achieved_alone": hog_duty(
                             out / "raw" / f"bw_hog_{tag}_alone.log"),
                         "duty_pct_achieved_during": hog_duty(hog_log), "phase": phase,
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

    return rows


def measure_cpu_util(out: Path, make_fn) -> list[dict]:
    """CPU utilisation per phase (§13.6); writes cpu_util.csv and raw/cpu_util_samples.csv."""
    # --- CPU utilisation per phase, in a SEPARATE run (§13.6)
    # Three scopes, because they answer different questions and disagree by 10x:
    # this thread's CPU time (what the Python loop itself burns), this process's
    # (adds the driver's worker threads), and the kernel's own per-core accounting
    # from /proc/stat (the only one that sees the whole machine).
    cpu_rows, sampler = [], CpuSampler(0.5)
    for phase in ("prefix", "ae", "full"):
        fn = make_fn(phase)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        sampler.start(phase)
        t0, c0, p0 = time.perf_counter(), time.thread_time(), time.process_time()
        n, win, prev = 0, [], (t0, c0, p0)
        while time.perf_counter() - t0 < 30.0:
            fn()
            n += 1
            now = time.perf_counter()
            if now - prev[0] >= 3.0:            # a distribution, not one number (§5)
                c, pr = time.thread_time(), time.process_time()
                win.append((100 * (c - prev[1]) / (now - prev[0]),
                            100 * (pr - prev[2]) / (now - prev[0])))
                prev = (now, c, pr)
        torch.cuda.synchronize()
        sampler.stop()
        wall = time.perf_counter() - t0
        for i, scope in enumerate(("thread", "process")):
            tot = (time.thread_time() - c0, time.process_time() - p0)[i]
            st = summarize([w_[i] for w_ in win]) if win else None
            cpu_rows.append({
                "phase": phase, "scope": scope, "pct": round(100 * tot / wall, 2),
                "median_pct": round(st["median_ms"], 2) if st else "",
                "p5_pct": round(st["p5_ms"], 2) if st else "",
                "p95_pct": round(st["p95_ms"], 2) if st else "",
                "n": st["n"] if st else n, "cores_busy": ""})
        cpu_rows += sampler.summary(phase)
        sysrow = next((r for r in cpu_rows
                       if r["phase"] == phase and r["scope"] == "system_all_cores"), None)
        print(f"  cpu_util {phase:8s} thread {cpu_rows[-len(sampler.summary(phase))-2]['pct']:6.2f} %"
              f"  system {sysrow['pct'] if sysrow else '?'} % "
              f"({sysrow['cores_busy'] if sysrow else '?'} cores)  n_iter={n}", flush=True)
    with (out / "cpu_util.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CPU_UTIL_FIELDS)
        w.writeheader()
        w.writerows(cpu_rows)
    with (out / "raw" / "cpu_util_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "t_s", "cpu", "busy_pct"])
        w.writeheader()
        w.writerows(sampler.rows)

    return cpu_rows


def measure_power(out: Path, make_fn) -> list[dict]:
    """Mean power per phase from the four INA3221 rails; writes power.csv."""
    # --- power per phase, from the four rails
    prows = []
    for phase in ("prefix", "ae", "full"):
        fn = make_fn(phase)
        acc = {}
        t_end = time.perf_counter() + 20
        while time.perf_counter() < t_end:
            fn()
            for rail, d in ina3221_rails().items():
                acc.setdefault(rail, []).append(d.get("mW") or 0)
        for rail, vals in acc.items():
            st = summarize(vals)
            prows.append({"phase": phase, "rail": rail,
                          "mean_mW": round(sum(vals) / len(vals), 1),
                          "median_mW": round(st["median_ms"], 1),
                          "p5_mW": round(st["p5_ms"], 1),
                          "p95_mW": round(st["p95_ms"], 1), "n": len(vals)})
        # the module total per sample, so its spread is the spread of the sum and
        # not the sum of four independently-taken spreads
        k = min(len(v) for v in acc.values())
        tot = [sum(v[i] for v in acc.values()) for i in range(k)]
        st = summarize(tot)
        prows.append({"phase": phase, "rail": "module",
                      "mean_mW": round(st["mean_ms"], 1),
                      "median_mW": round(st["median_ms"], 1),
                      "p5_mW": round(st["p5_ms"], 1),
                      "p95_mW": round(st["p95_ms"], 1), "n": k})
    with (out / "power.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "rail", "mean_mW", "median_mW",
                                          "p5_mW", "p95_mW", "n"])
        w.writeheader()
        w.writerows(prows)

    return prows


def emit_rows(mi, rows, cpu_rows, prows, style: str) -> None:
    """The model_inputs.csv rows E11e is responsible for."""
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
        if c["scope"].startswith("cpu"):
            continue                       # per-core detail stays in cpu_util.csv
        mi.add("cpu_util", c["pct"], "pct", "value", c["n"], "cpu_util.csv",
               key={"prefix": "vlm_prefix", "ae": "ae_total", "full": "e2e"}[c["phase"]],
               variant=f"scope={c['scope']}", exec_style=style)
    for pr in prows:
        mi.add("p_mean_W", pr["mean_mW"] / 1000.0, "W", "mean", pr["n"], "power.csv",
               key=f"{pr['phase']}/{pr['rail']}", exec_style=style)
        mi.add("p_mean_W", pr["median_mW"] / 1000.0, "W", "median", pr["n"],
               "power.csv", key=f"{pr['phase']}/{pr['rail']}", exec_style=style)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="pi0 | pi05 | smolvla. T011 extends E11 beyond pi0; the expert layout and the checkpoint-specific processors come from bench/model_registry.py.")
    ap.add_argument("--outdir-prefix", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--modes", default="eager,reduce-overhead")
    ap.add_argument("--phases", default="prefix")
    args = ap.parse_args()
    spec = MR.get(args.model)
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5})
    except OSError as e:                                           # noqa: BLE001
        print(f"  could not pin foreground: {e}")

    for mode in [m for m in args.modes.split(",") if m]:
        out = Path(f"{args.outdir_prefix}_{mode}_mempressure")
        (out / "raw").mkdir(parents=True, exist_ok=True)
        hog_self_test(out)          # before the model loads and the 47 minutes start
        cm = None if mode == "eager" else mode
        policy, cfg, shipped = MR.load_policy(spec, n=args.steps, compile_mode=cm)
        real = P.assert_real_weights(policy, spec=spec)
        fx = P.load_fixture()
        batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
        # P and S as THIS model builds them. The literals 816 and 51 that used to be
        # written here are pi0's, and would have been written unchanged for pi0.5 and
        # SmolVLA -- a wrong constant in the artefact the PIM model consumes.
        ps = PB.measure_ps(policy, spec, batch)
        print(f"\n=== {mode} ===", flush=True)
        def make_fn(phase, _p=policy, _b=batch):
            return PL.make_phase(_p, _b, phase, args.steps, spec)
        rows = measure_loads(out, make_fn, [p for p in args.phases.split(",") if p], args.seconds)
        cpu_rows = measure_cpu_util(out, make_fn)
        prows = measure_power(out, make_fn)

        mi = ModelInputs(spec.name, MR.resolved_commit(spec), sh_out(f"git -C {REPO} rev-parse HEAD"),
                         "MAXN+jetson_clocks", ps["P"], ps["S"], args.steps)
        style = "eager" if cm is None else "compiled"
        emit_rows(mi, rows, cpu_rows, prows, style)
        n_mi = mi.write(out / "model_inputs.csv")

        for f_ in sorted((out / "raw").glob("latency_samples_*.csv")):
            compress_and_record(f_)
        sw = software_versions()
        write_json(out / "meta.json", {
            "experiment": "E11e", "task": "T010",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": spec.name, "model_repo": spec.checkpoint,
            "model_commit": MR.resolved_commit(spec),
        "model_revision": spec.revision or "not pinned (hub default branch; see model_commit)",
            "checkpoint": spec.checkpoint, "precision": "bfloat16",
            "compile_mode": cm, "batch": 1, "num_denoise_steps": args.steps,
            "action_horizon": getattr(cfg, "chunk_size", None),
            "prefix_tokens": ps["P"], "suffix_tokens": ps["S"],
            "token_counts": ps,
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
        e11e(out)
        meta["raw_files"] = collect_raw(out, meta["command"])
        write_json(out / "meta.json", meta)
        print(f"wrote {out} ({n_mi} rows)", flush=True)
        del make_fn, policy       # make_fn holds the policy through its defaults
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
