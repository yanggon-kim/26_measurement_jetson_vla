"""E12 Step 5: energy of the pipelined schedule against the serial one, per protocol §4.

WHAT THIS CAN AND CANNOT SHOW, STATED FIRST
E12 measured `T_pipe` = `A_iso` + `B_iso`: the pipeline is serial, because the host issues for
100 % of the wall clock (issue/wall = 1.0000). So the *expected* result here is that the
pipelined schedule costs the same energy per frame as the serial one, and this run exists to
confirm that rather than to discover a saving. A measured "no difference" is the deliverable
Step 5 asks for; if a difference DID appear it would be evidence of something the timing
harness missed, which is why it is worth the 3 minutes of rails.

METHOD, following E7 (protocol §4) so the numbers are comparable to it
  * 60 s idle baseline measured IN-SESSION, not borrowed from another run;
  * >= 60 s pipelined loop and >= 60 s serial loop;
  * four rails from sysfs at 20 Hz (tegrastats omits the memory rail on this board) plus a
    tegrastats log alongside;
  * the serial arm is a WHOLE-FRAME loop -- prefix and expert back to back through
    `corun.run_serial` -- and is NOT reconstructed by adding a prefix-only loop to an
    expert-only loop, which in E7 do not sum to `full`.

Both arms run the identical frame list through the identical stage functions
(`corun.run_pipeline` / `corun.run_serial`), so the only difference between them is the
schedule. Energy per frame is reported above idle as well as absolute, because the idle draw
of an Orin dominates a short-frame workload and an absolute-only number would understate the
schedule's share.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import corun as CR             # noqa: E402
import pi0_common as P         # noqa: E402
import power_core as PC        # noqa: E402
import power_phases as PP      # noqa: E402

MEM_RAIL = PP.MEM_RAIL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=10,
                    help="frames per timed call. A pipeline has a start-up and a drain, so "
                         "one frame per call would measure those rather than the steady "
                         "state; 10 amortises them while keeping each call short.")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--nosync", action="store_true", default=True)
    ap.add_argument("--fixtures", default="bench/fixtures/e12")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty -- protocol §5 forbids overwriting")
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print(f"  gate: power_mode={gate['power_mode']} sm_count={gate['sm_count']}", flush=True)

    # Idle FIRST and in-session, before the model is loaded: an idle baseline taken after a
    # 15 GB model is resident is not this device's idle.
    print(f"  {args.seconds:.0f} s idle baseline (protocol §4) ...", flush=True)
    idle_means, _, idle_s = PP.measure(None, args.seconds, "idle",
                                       out / "raw" / "tegrastats_idle.log")
    print("    idle " + "  ".join(f"{k}={v:.0f}mW" for k, v in idle_means.items()),
          flush=True)

    import model_registry as MR                                        # noqa: PLC0415
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    nosync_note = CR.patch_pi0_nosync(policy) if args.nosync else "stock pi0"
    print(f"  {nosync_note}", flush=True)

    frames, frames_manifest = CR.load_frames(policy, spec, REPO / args.fixtures,
                                             args.frames, seed0=0)
    sp, se = torch.cuda.Stream(), torch.cuda.Stream()

    def pipelined():
        CR.run_pipeline(frames, args.steps, sp, se)

    def serial():
        CR.run_serial(frames, args.steps)

    for _ in range(2):                      # warm both schedules before either is measured
        pipelined()
        serial()
    torch.cuda.synchronize()

    rows, all_samples = [], [{"phase": "idle", **x} for x in idle_s]
    rails = sorted(idle_means)
    results = {}
    for phase, fn in (("pipelined", pipelined), ("serial_full", serial)):
        print(f"  {args.seconds:.0f} s {phase} loop ...", flush=True)
        means, r, samples = PP.measure(fn, args.seconds, phase,
                                       out / "raw" / f"tegrastats_{phase}.log")
        all_samples += [{"phase": phase, **x} for x in samples]
        # r["median_ms"] times ONE CALL, which is `--frames` frames.
        per_frame_ms = r["median_ms"] / args.frames
        module = sum(means.values())
        module_idle = sum(idle_means.values())
        no_mem = sum(v for k, v in means.items() if k != MEM_RAIL)
        no_mem_idle = sum(v for k, v in idle_means.items() if k != MEM_RAIL)
        results[phase] = {
            "call_ms_median": r["median_ms"], "frames_per_call": args.frames,
            "per_frame_ms": round(per_frame_ms, 4),
            "module_mW": round(module, 1), "module_above_idle_mW": round(module - module_idle, 1),
            "energy_per_frame_mJ": round(module * per_frame_ms / 1e3, 4),
            "energy_per_frame_above_idle_mJ":
                round((module - module_idle) * per_frame_ms / 1e3, 4),
        }
        print(f"    {phase}: {per_frame_ms:.2f} ms/frame  {module:.0f} mW  "
              f"({module - module_idle:+.0f} mW above idle)  "
              f"{results[phase]['energy_per_frame_mJ']:.2f} mJ/frame", flush=True)
        for rail in rails + ["module", "module_without_memory_rail"]:
            mw = {"module": module, "module_without_memory_rail": no_mem}.get(
                rail, means.get(rail))
            base = {"module": module_idle,
                    "module_without_memory_rail": no_mem_idle}.get(rail, idle_means.get(rail))
            rows.append({
                "model": "pi0", "mode": "eager", "mechanism":
                    "pi0_nosync" if args.nosync else "stock",
                "phase": phase, "rail": (f"{rail} (regulator input)"
                                         if rail == MEM_RAIL else rail),
                "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
                "N": args.steps, "frames_per_call": args.frames,
                "mean_mW": round(mw, 2), "idle_mW": round(base, 2),
                "above_idle_mW": round(mw - base, 2),
                "latency_ms_per_frame": round(per_frame_ms, 4),
                "energy_per_frame_mJ": round(mw * per_frame_ms / 1e3, 4),
                "energy_per_frame_above_idle_mJ":
                    round((mw - base) * per_frame_ms / 1e3, 4),
                "n_samples": len(samples)})

    PC.write_power_csv(out / "power.csv", rows, all_samples)

    p, s = results["pipelined"], results["serial_full"]
    ratio_t = p["per_frame_ms"] / s["per_frame_ms"]
    ratio_e = p["energy_per_frame_mJ"] / s["energy_per_frame_mJ"]
    verdict = (
        f"pipelined / serial = {ratio_t:.4f} in time, {ratio_e:.4f} in energy per frame. "
        + ("Neither differs materially, which is what T_pipe = A_iso + B_iso predicts: the "
           "schedule does not overlap, so it cannot save either. "
           if 0.95 <= ratio_t <= 1.05 and 0.95 <= ratio_e <= 1.05 else
           "*** These differ by more than 5 %, which the timing harness did not predict. "
           "Do not report this as a Step 5 confirmation until the difference is explained. ")
        + "Energy here is module power x measured frame time; it is not a separate claim "
          "about efficiency beyond what those two measurements carry.")
    print(f"\n  {verdict}", flush=True)

    meta = {
        "experiment": "E12 Step 5 -- energy of the pipelined vs the serial schedule",
        "method": "protocol §4, as E7: in-session idle baseline, >=60 s per arm, four sysfs "
                  "rails at 20 Hz plus tegrastats. The serial arm is a whole-frame loop via "
                  "corun.run_serial, NOT a prefix loop plus an expert loop, which in E7 do "
                  "not sum to `full`.",
        "model": "pi0", "rung": "eager", "N": args.steps,
        "mechanism": "pi0_nosync" if args.nosync else "stock", "nosync_note": nosync_note,
        "frames_per_call": args.frames, "seconds_per_arm": args.seconds,
        "frames_manifest": frames_manifest,
        "gate": gate, "results": results, "verdict": verdict,
        "command": " ".join([sys.executable, *sys.argv]),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      capture_output=True, text=True).stdout.strip(),
        "checkpoint": spec.checkpoint, "revision": spec.revision,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "idle_means_mW": {k: round(v, 2) for k, v in idle_means.items()},
        "measured": ["rail power per arm", "frame time per arm", "idle rail power"],
        "derived": ["energy per frame = module mW x frame ms",
                    "energy above idle = (module - idle) mW x frame ms",
                    "the two ratios in `verdict`"],
        "not_measured": [
            "Energy of a pipeline that ACTUALLY overlaps -- none exists on this device, see "
            "the report's §1 and §5. This arm measures the schedule as it really runs.",
            "Per-stage energy attribution within a frame: the rails are sampled at 20 Hz and "
            "a frame is ~670 ms, so a single frame carries ~13 samples -- enough for a frame "
            "mean, not for splitting prefix from expert inside one.",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {out / 'power.csv'} and meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
