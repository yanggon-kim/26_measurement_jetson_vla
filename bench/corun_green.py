"""E12 Step 3: pi0's SM-sensitivity curve, measured with green contexts on one Orin GPU.

THE QUESTION THIS ANSWERS, AND WHY IT IS THE RIGHT ONE LEFT
E12's headline is that pi0's prefix and expert do not overlap on one GPU, and that the reason
is the HOST: issue/wall = 1.0000. So the co-run question -- KerColle's head-of-line blocking
at the CTA dispatcher, found in a simulator of 128-170 SM desktop GPUs -- was never actually
put to Orin's 16 SMs. The direct routes are closed: one Python thread cannot feed two streams
(§13.7), the compiled rung corrupts across streams, and two processes are OOM-killed (Orin's
62 GB is shared CPU+GPU; one pi0 at bf16_both is ~15 GB).

Green contexts partition SMs inside ONE process, which dodges the memory limit and turns the
question around:

    How much slower is ONE stage when it is given 8 SMs instead of 16?

The logic is that of a necessary condition. Contention between two stages requires that each
stage can actually USE the SMs the other would take. If the expert runs at nearly the same
speed on 8 SMs as on 16, it was not using 16, and a co-runner could have occupied the other 8
for free -- so contention cannot be what limits a co-run on this device. If instead a stage
scales close to ideal, it is SM-hungry and contention is a real hazard. This is weaker than a
two-stage co-run measurement in one respect (it does not exhibit interference directly) and
stronger in another (it is not confounded by the host bottleneck that defeated every direct
attempt, and it isolates each stage's own demand).

WHAT MUST HOLD FOR ANY OF IT TO COUNT, both checked in this run and recorded in its own files
  1. The partition is real. bench/green_ctx.verify_partition_is_real() times a compute-bound
     GEMM per SM count; a flat curve means the green context is decorative and every number
     here would be plausible milliseconds measuring nothing. Refuses to time if flat.
  2. The numerics are unchanged. Green contexts partition SMs but share the device address
     space with torch's primary context, so the caching allocator's tensors should be reachable
     -- SHOULD BE is not IS, and a silent numerical difference would invalidate everything, so
     actions computed on the green stream are compared against the default stream bitwise.

Binding: protocol §14, §1, §5; spec docs/E12_corun/README.md §5 Step 3.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import corun as CR              # noqa: E402  the same stage functions the rest of E12 uses
import green_ctx as GC          # noqa: E402
import phase_builders as PB     # noqa: E402
import pi0_common as P          # noqa: E402


def summarize(v: list[float]) -> dict:
    s = sorted(v)
    return {"n": len(v), "median_ms": round(statistics.median(s), 4),
            "mean_ms": round(statistics.fmean(s), 4),
            "p5_ms": round(s[max(0, int(0.05 * len(s)) - 1)], 4),
            "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
            "min_ms": round(s[0], 4), "max_ms": round(s[-1], 4)}


def time_on(fn, g, seconds: float, min_iters: int, warm: int) -> list[float]:
    """Time `fn` inside green context `g` (None = the whole device).

    A time budget rather than a fixed iteration count: at 4 SMs a stage takes roughly 3x
    longer, and a fixed count would let the low-SM points dominate the session's length.

    Uses `g.activate()`, which makes the green context CURRENT as well as selecting its
    stream. Both are required -- with only the stream wrapped, cuBLASLt's fused addmm fails
    on a green stream and torch's fallback leaves an invalid context (measured; see
    green_ctx.GreenCtxStream.activate).
    """
    cm = g.activate() if g is not None else contextlib.nullcontext()
    samples: list[float] = []
    with cm:
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds or len(samples) < min_iters:
            a = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - a) * 1e3)
            if len(samples) >= min_iters and time.perf_counter() - t0 > seconds * 3:
                break                      # a pathologically slow point cannot stall the run
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="eager",
                    help="eager matches E12's published MAXN run, so the 16-SM point is "
                         "directly comparable to it. bf16_both is the rung that loads the "
                         "GPU (372 kernels/step, 47 %% idle) and is where SM demand should "
                         "be highest if it exists anywhere.")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--sms", default="4,6,8,10,12,16")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--min-iters", type=int, default=30)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--fixtures", default="bench/fixtures/e12")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty -- protocol §5 forbids overwriting")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    gate = P.gate()
    print(f"  gate: power_mode={gate['power_mode']} sm_count={gate['sm_count']}")

    # --- control 1: is the partition real at all? Refuse to time if not.
    print("  control 1: verifying green contexts actually confine work ...")
    ver = GC.verify_partition_is_real(sm_counts=(4, 8, 16))
    (out / "partition_verification.json").write_text(json.dumps(ver, indent=2) + "\n")
    for p in ver["points"]:
        print(f"    {p['granted_sm']:2d} SMs: {p['ms_per_gemm']:8.4f} ms/gemm  "
              f"slowdown {p.get('slowdown_vs_full', 0):.3f}x "
              f"(ideal {p.get('ideal_slowdown', 0):.3f}x)")
    print(f"    {ver['verdict']}")
    if not ver["partition_is_real"]:
        (out / "REFUSED.md").write_text(
            "# No timings were taken\n\n"
            "`green_ctx.verify_partition_is_real()` found a FLAT GEMM curve across SM "
            "counts: the green context did not restrict execution. Per-SM-count stage times "
            "from an unrestricted partition would be plausible milliseconds measuring "
            "nothing, so none were recorded.\n\n"
            f"```json\n{json.dumps(ver, indent=2)}\n```\n")
        print("  REFUSING to time: partition is not real. See REFUSED.md")
        return 3

    import model_registry as MR                                        # noqa: PLC0415
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    rung_note = "eager"
    if args.rung and args.rung != "eager":
        import opt_ladder as OL                                        # noqa: PLC0415
        compiled, prep, prec, comp_s, fails, _ = OL.apply_rung(policy, spec, args.rung)
        if fails:
            raise SystemExit(f"rung {args.rung} failed to compile: {fails}")
        rung_note = f"{args.rung}: prep=[{prep}] compiled={compiled} matmul={prec}"
        print(f"  {rung_note} in {comp_s:.1f} s")

    fixture = json.loads((REPO / args.fixtures / "obs_0.json").read_text())
    batch, _, _ = P.build_batch(policy, fixture, fixtures=REPO / args.fixtures,
                                checkpoint=spec.checkpoint, revision=spec.revision)
    ctx = PB.build_ctx(policy, spec, batch)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda",
                        dtype=torch.float32)
    pkv, pad = CR.prefix_stage(ctx)

    def prefix():
        CR.prefix_stage(ctx)

    def expert():
        CR.expert_stage(ctx, pkv, pad, noise, args.steps)

    # --- control 2: do the numerics survive a green-context stream?
    print("  control 2: comparing actions on a green stream against the default stream ...")
    g8 = GC.GreenCtxStream(8)
    with torch.no_grad():
        ref = CR.expert_stage(ctx, pkv, pad, noise, args.steps).clone()
        torch.cuda.synchronize()
        with g8.activate():
            got = CR.expert_stage(ctx, pkv, pad, noise, args.steps).clone()
    d = (ref - got).abs().max().item()
    nan = bool(torch.isnan(got).any().item())
    equiv = {"max_abs_diff": d, "bitwise_identical": d == 0.0, "any_nan": nan,
             "granted_sm": g8.granted_sm,
             "requirement": "eager on a partitioned GPU must be bitwise identical: the same "
                            "kernels run, only on fewer SMs. A non-zero difference means the "
                            "green stream is not computing what the default stream computes, "
                            "and no timing from it would describe pi0.",
             "passed": (d == 0.0) and not nan}
    (out / "equivalence.json").write_text(json.dumps(equiv, indent=2) + "\n")
    print(f"    max|d| = {d:.3e}  nan={nan}  -> "
          f"{'PASS' if equiv['passed'] else '*** FAIL ***'}")
    g8.close()
    if not equiv["passed"] and args.rung == "eager":
        (out / "REFUSED.md").write_text(
            "# No timings were taken\n\nActions computed on a green-context stream differ "
            f"from the default stream (max|d| = {d:.6e}, nan={nan}). In eager the same "
            "kernels run on fewer SMs and the result must be bitwise identical, so this is a "
            "correctness failure in the mechanism, not a numerical tolerance question.\n")
        print("  REFUSING to time: green-stream numerics differ. See REFUSED.md")
        return 4

    # --- the measurement
    sms = [int(x) for x in args.sms.split(",")]
    full = GC.device_sm_count()
    rows = []
    for want in sms:
        g = None if want >= full else GC.GreenCtxStream(want)
        granted = full if g is None else g.granted_sm
        for label, fn in (("prefix", prefix), ("expert", expert)):
            s = summarize(time_on(fn, g, args.seconds, args.min_iters, args.warm))
            rows.append({"stage": label, "want_sm": want, "granted_sm": granted,
                         "rung": args.rung, "N": args.steps,
                         "power_mode": gate["power_mode"], **s})
            print(f"    {label:6s} {granted:2d} SMs: median {s['median_ms']:8.2f} ms  "
                  f"(p5 {s['p5_ms']:7.2f}  p95 {s['p95_ms']:7.2f}  n={s['n']})")
        if g is not None:
            g.close()

    # derived: slowdown vs the full-GPU point of THIS session, per stage
    base = {r["stage"]: r["median_ms"] for r in rows if r["granted_sm"] == full}
    for r in rows:
        b = base.get(r["stage"])
        r["slowdown_vs_16sm"] = round(r["median_ms"] / b, 4) if b else ""
        r["ideal_slowdown"] = round(full / r["granted_sm"], 4)
        r["sm_efficiency"] = (round((r["slowdown_vs_16sm"] - 1) /
                                    (r["ideal_slowdown"] - 1), 4)
                              if b and r["ideal_slowdown"] > 1 else "")

    csv_p = out / "sm_sensitivity.csv"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {csv_p}")

    # the interpretation, printed so it cannot be read off the CSV without the caveat
    print("\n  --- what this says about co-running (per stage, 8 SMs vs 16) ---")
    for stage in ("prefix", "expert"):
        r8 = [r for r in rows if r["stage"] == stage and r["granted_sm"] == 8]
        if not r8:
            continue
        sd, eff = r8[0]["slowdown_vs_16sm"], r8[0]["sm_efficiency"]
        verdict = ("SM-HUNGRY: it uses the SMs it is given, so a co-runner taking half "
                   "WOULD cost it -- contention is a real hazard for this stage"
                   if sd >= 1.5 else
                   "SM-INSENSITIVE: losing half the SMs costs it little, so it was not "
                   "using them and a co-runner could have had them nearly for free")
        print(f"  {stage:6s} 8 SMs is {sd:.3f}x its 16-SM time (ideal 2.000x, "
              f"efficiency {eff}) -> {verdict}")

    meta = {
        "experiment": "E12 Step 3 -- SM partitioning via CUDA green contexts",
        "mechanism": "green_contexts_one_process",
        "model": "pi0", "rung": args.rung, "rung_note": rung_note, "N": args.steps,
        "gate": gate, "device_sm_count": full, "sm_points": sms,
        "command": " ".join([sys.executable, *sys.argv]),
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      capture_output=True, text=True).stdout.strip(),
        "checkpoint": spec.checkpoint, "revision": spec.revision,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "driver_green_ctx": "cuGreenCtxCreate + cuGreenCtxStreamCreate via ctypes, driver 12.6",
        "controls": {"partition_verification": ver["verdict"],
                     "equivalence": equiv},
        "measured": ["per-stage wall time per SM count (host-side, with a full sync per "
                     "iteration, matching the rest of E12)"],
        "derived": ["slowdown_vs_16sm = median / this session's 16-SM median, per stage",
                    "ideal_slowdown = 16 / granted_sm",
                    "sm_efficiency = (slowdown-1)/(ideal_slowdown-1): 1.0 means the stage "
                    "scales exactly with SM count, 0.0 means it is completely insensitive"],
        "not_measured": [
            "Two stages running CONCURRENTLY in separate partitions. This run establishes "
            "each stage's own SM demand, which is the necessary condition for contention; it "
            "does not exhibit interference. Running both at once in two green contexts is the "
            "natural next step and needs the host-issue problem solved first, since one "
            "Python thread still issues both (§13.7 forbids a second).",
            "nsys timeline per partition.",
            "Whether the driver's SM partition is spatially contiguous (TPC-aligned) -- the "
            "split granted (4,4,4,4)/(6,6)/(8,8) suggests TPC pairs, but this was not "
            "verified against hardware IDs.",
        ],
        "caveat": (
            "sm_efficiency below ~0.3 for a stage means that stage cannot be the victim OR "
            "the cause of CTA-level contention on this device, because it does not use the "
            "SMs it already has. That is a statement about Orin's 16 SMs and pi0's kernel "
            "shapes, NOT a refutation of KerColle, which modelled 128-170 SM GPUs where the "
            "same model's grids may well exceed capacity."),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {out / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
