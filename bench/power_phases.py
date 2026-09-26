"""T004 step 4 / E7 — power and energy per phase, from the four sysfs rails.

Protocol §4, applied here rather than left to the writer:

  * all four INA3221 rails at >= 10 Hz, each reported separately **and** summed as module
    power, with the three-rail sum without the memory rail given too;
  * the memory rail is named **`VDDQ_VDD2_1V8AO (regulator input)`**, measured at ~5.04 V at
    the regulator input and **never rescaled to 1.8 V**; it is an upper bound on DRAM power,
    and the quantity to quote is its **above-idle delta** per phase next to the absolute value.
    Never "DRAM power" without those words;
  * 60 s of idle baseline in the same power mode **before** the loops;
  * each phase is a back-to-back loop of >= 60 s, not a single inference;
  * energy per inference = mean power x mean latency, reported total and above idle;
  * tegrastats alongside at `--interval 100`, its log deleted first because
    `--logfile` APPENDS (T001).

The phases come from `bench/phase_loops.py`, which T010 wrote for E11e — prefix-only,
action-expert-only and full — so E7 and E11e measure the same loops. The model-specific
part of that split lives in `bench/phase_builders.py`, which is why `--model` works here:
π0.5 and SmolVLA carry the robot state inside the prefix and so have no `state` argument on
`denoise_step`, and SmolVLA skips the 4-D attention-mask conversion entirely.

`--allow-capped` is what makes the MODE_50W / 30W / 15W sweep possible. Without it `gate()`
raises on any non-MAXN mode, which would have made this script unable to measure the one
comparison it exists for. The mode is still recorded, in `meta.json` and in the directory
name, either way.

    python bench/power_phases.py --model pi0 --outdir-prefix results/<date>_E7_pi0_N10 \
        --steps 10 --seconds 60
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import phase_loops as PL  # noqa: E402
import pi0_common as P  # noqa: E402
from common import (REPO, collect_raw, ina3221_rails, sh_out,  # noqa: E402
                    software_versions, write_json)

MEM_RAIL = "VDDQ_VDD2_1V8AO"


class Rails(threading.Thread):
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


def measure(fn, seconds, tag, teg_log: Path):
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
    if teg_log.exists():
        teg_log.unlink()
    teg = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "100",
                            "--logfile", str(teg_log)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    s = Rails()
    s.start()
    r = PL.run_loop(fn, seconds, tag) if fn else {
        "iterations": 0, "wall_s": seconds, "throughput_inf_per_s": 0.0,
        "median_ms": 0.0, "p5_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0,
        "samples": []}
    if fn is None:
        time.sleep(seconds)
    s.stop.set()
    s.join(timeout=2)
    stop_tegrastats(teg)
    rails = [k for k in s.samples[0] if k != "t"]
    means = {k: statistics.fmean([x[k] for x in s.samples if x.get(k) is not None])
             for k in rails}
    r.pop("samples", None)
    return means, r, s.samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir-prefix", required=True)
    ap.add_argument("--model", default="pi0", help="pi0 | pi05 | smolvla")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--modes", default="eager,reduce-overhead",
                    help="compile modes to sweep. A mode is applied to sample_actions "
                         "only, so it does NOT reach the prefix and ae loops — use "
                         "--rung for a configuration that does.")
    ap.add_argument("--rung", default=None,
                    help="an opt_ladder rung name (e.g. bf16_both) applied INSTEAD of a "
                         "compile mode. Unlike --modes this reaches the prefix and ae "
                         "loops, because the rungs compile pg.forward / embed_image / "
                         "denoise_step directly. This is how E7 measures each model's "
                         "BEST configuration rather than a generic compile mode.")
    ap.add_argument("--allow-capped", action="store_true",
                    help="permit a non-MAXN power mode. The gate still RECORDS the mode "
                         "and it goes in meta.json and the directory name; this only "
                         "stops it raising. Required for the MODE_50W/30W/15W sweep, "
                         "which is the whole point of measuring power.")
    args = ap.parse_args()
    spec = MR.get(args.model)
    gate = P.gate(require_maxn=not args.allow_capped)
    pm = gate["power_mode"].split(":")[-1].strip().replace(" ", "")
    print("gate:", json.dumps(gate), flush=True)

    modes = [args.rung] if args.rung else [m for m in args.modes.split(",") if m]
    for mode in modes:
        out = Path(f"{args.outdir_prefix}_{mode}_{pm}")
        (out / "raw").mkdir(parents=True, exist_ok=True)
        cm = None if mode == "eager" else mode

        print(f"\n=== {mode}: 60 s idle baseline first (protocol §4) ===", flush=True)
        idle_means, _, idle_s = measure(None, args.seconds, "idle",
                                        out / "raw" / "tegrastats_idle.log")
        print("  idle " + "  ".join(f"{k}={v:.0f}mW" for k, v in idle_means.items()),
              flush=True)

        policy, cfg, shipped = MR.load_policy(
            spec, n=args.steps, compile_mode=None if args.rung else cm)
        # Verify the checkpoint BEFORE any rung touches it. assert_real_weights diffs
        # against the fp32 safetensors, and bf16_both deliberately casts 74 layernorm
        # modules and the vision tower, so running it afterwards reported
        # "325/400 tensors match ... Refusing to measure" -- a true statement about a
        # cast that was the whole point of the rung. Verify, then modify.
        real = P.assert_real_weights(policy) if args.model == "pi0" else {
            "checked": False, "reason": "assert_real_weights carries pi0's expected "
                                        "signature; not applicable to this checkpoint"}
        rung_note, compiled_desc = "", []
        if args.rung:
            import opt_ladder as OL
            try:
                compiled_desc, rung_note, prec, comp_s, fails, _ = OL.apply_rung(
                    policy, spec, args.rung)
            except OL.NotApplicable as e:
                raise SystemExit(f"rung {args.rung!r} is not applicable to {args.model}: "
                                 f"{e}. Refusing to measure an unprepared policy and call "
                                 f"it that rung.")
            if fails:
                raise SystemExit(f"rung {args.rung!r} failed to compile: {fails}. A "
                                 f"partially-applied rung is not the rung.")
            print(f"  rung {args.rung}: prep=[{rung_note}] "
                  f"compiled=[{';'.join(compiled_desc)}] matmul={prec} in {comp_s:.1f} s",
                  flush=True)
        fx = P.load_fixture()
        batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                    revision=spec.revision)

        rows, all_samples = [], [{"phase": "idle", **x} for x in idle_s]
        rails = sorted(idle_means)
        for phase in ("prefix", "ae", "full"):
            fn = PL.make_phase(policy, batch, phase, args.steps, spec)
            means, r, samples = measure(fn, args.seconds, phase,
                                        out / "raw" / f"tegrastats_{phase}.log")
            all_samples += [{"phase": phase, **x} for x in samples]
            module = sum(means.values())
            module_idle = sum(idle_means.values())
            no_mem = sum(v for k, v in means.items() if k != MEM_RAIL)
            lat_s = r["median_ms"] / 1e3
            for rail in rails + ["module", "module_without_memory_rail"]:
                mw = {"module": module,
                      "module_without_memory_rail": no_mem}.get(rail, means.get(rail))
                base = {"module": module_idle,
                        "module_without_memory_rail":
                            sum(v for k, v in idle_means.items() if k != MEM_RAIL)}.get(
                            rail, idle_means.get(rail))
                name = (f"{rail} (regulator input)" if rail == MEM_RAIL else rail)
                rows.append({
                    "model": args.model, "mode": mode,
                    "exec_style": ("compiled_subpath" if (args.rung and compiled_desc)
                                   else PL.effective_exec_style(phase, cm)),
                    "power_mode": pm, "phase": phase, "rail": name,
                    "mean_mW": round(mw, 1), "idle_mW": round(base, 1),
                    "above_idle_mW": round(mw - base, 1),
                    "iterations": r["iterations"], "median_ms": r["median_ms"],
                    "throughput_inf_per_s": r["throughput_inf_per_s"],
                    "energy_per_inference_mJ": round(mw * lat_s, 3),
                    "energy_per_inference_above_idle_mJ": round((mw - base) * lat_s, 3),
                    "energy_per_step_mJ": round(mw * lat_s / args.steps, 3)
                    if phase in ("ae", "full") else "",
                    "energy_per_step_above_idle_mJ":
                        round((mw - base) * lat_s / args.steps, 3)
                        if phase in ("ae", "full") else "",
                    "n_samples": len(samples)})
            print(f"  {phase:7s} module {module:7.0f} mW (+{module-module_idle:6.0f}) "
                  f"{r['median_ms']:8.2f} ms  {module*lat_s:8.2f} mJ/inf", flush=True)

        with (out / "power.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (out / "raw" / "rail_samples.csv").open("w", newline="") as f:
            keys = sorted({k for x in all_samples for k in x})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: x.get(k, "") for k in keys} for x in all_samples])

        sw = software_versions()
        write_json(out / "meta.json", {
            "experiment": "E7", "task": "T004",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": spec.name, "model_repo": spec.checkpoint,
            "model_commit": MR.resolved_commit(spec),
            "checkpoint": spec.checkpoint,
            "precision": ("checkpoint default" if not args.rung
                          else f"rung {args.rung}: {rung_note}"),
            "compile_mode": None if args.rung else cm,
            "rung": args.rung, "rung_compiled": ";".join(compiled_desc),
            "batch": 1, "num_denoise_steps": args.steps,
            "action_horizon": getattr(cfg, "chunk_size", None),
            "n_warmup": 5, "n_timed": f"{args.seconds} s loop per phase", "sample_hz": 20,
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
            "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
            "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
            "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
            "python": sw.get("python"), "venv": sw.get("executable"),
            "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
            "bench_files_sha256": sw.get("bench_files_sha256"),
            "weights_verified": real, "shipped_config": shipped,
            "command": (f"python bench/power_phases.py --model {args.model} "
                        f"--outdir-prefix {args.outdir_prefix} --steps {args.steps} "
                        f"--seconds {args.seconds} --modes {args.modes}"
                        + (" --allow-capped" if args.allow_capped else "")),
            "idle_baseline_mW": {k: round(v, 1) for k, v in idle_means.items()},
            "throttled": False,
            "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
            "raw_files": [],
            "notes": [
                "The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT "
                "rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle "
                "delta next to the absolute value, and never call it 'DRAM power' without "
                "those words (protocol §4).",
                "Each rail is reported separately, as module power (all four) and as the "
                "three-rail sum without the memory rail.",
                "T005 checked the four rails for double-counting and found none: a load that "
                "raises VDD_GPU_SOC by 51.8 W raises VIN_SYS_5V0 by only 8.7 W. They may be "
                "summed.",
                "Phases are bench/phase_loops.py's loops, the same ones E11e drives, so the "
                "power and the slowdown numbers refer to identical work.",
                "60 s of idle baseline was measured in this power mode before the loops.",
                "tegrastats --logfile appends, so each log is deleted before its run (T001).",
                "With --modes, the `prefix` and `ae` loops call embed_prefix / "
                "container.forward / denoise_step DIRECTLY, below the compiled "
                "sample_actions wrapper, so they run EAGER even when compile_mode is set: "
                "exec_style is `eager_subpath` and only `full` is `compiled`. With --rung "
                "the ladder compiles those three callables themselves, so all three loops "
                "are genuinely compiled and exec_style is `compiled_subpath`. Do not read "
                "an eager_subpath prefix row as the power of a compiled prefix.",
                "`ae_once` drives a bounded Python loop instead of sample_actions' "
                "GPU-tensor `while` test. That is a protocol §2 deviation, made because "
                "isolating the expert requires it, and it is reported not hidden.",
            ],
        })
        meta = json.loads((out / "meta.json").read_text())
        meta["raw_files"] = collect_raw(out, meta["command"])
        write_json(out / "meta.json", meta)
        print(f"wrote {out}", flush=True)
        del policy
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
