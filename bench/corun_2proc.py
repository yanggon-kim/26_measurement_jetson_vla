"""E12 Step 3: the two stages in two PROCESSES, which is where GPU contention can appear.

Specification `docs/E12_corun/README.md` §5 Step 3 ("two processes, no MPS — the naive
multi-process deployment"). Binding: protocol §14, §1, §5.

## Why this exists, and why it is not a consolation prize

The in-process two-stream pipeline (`bench/corun.py`) established two blockers:

  1. **π0 eager is entirely CPU-issue-bound** — measured `issue/wall` = 1.0000 over a
     99.5-second run, consistent with E5's 84.2 % GPU idle over 968 kernels/step. One Python
     thread cannot keep two streams fed, so nothing overlaps and **GPU contention cannot
     arise at all**. KerColle's mechanism is contention; a configuration that never fills the
     GPU cannot exhibit it.
  2. **The compiled `bf16_both` rung corrupts when pipelined across two streams** —
     non-deterministically, max|d| 1.273 on one run and all-NaN on another, silently.

Two processes clears both at once, which is the point:

  * Each process has its **own Python interpreter**, so the two stages are issued by two
    CPU cores in parallel. The issue bottleneck stops being shared.
  * Each process has **one sequential call stream**, so torch.compile's CUDAGraphTrees never
    sees two streams. The compiled rung becomes usable — and the compiled rung is the one
    with 372 kernels/step and 47 % idle, i.e. the one that actually loads the GPU.

So this is the first configuration in E12 where the question "do the two stages contend on
one Orin GPU?" can be asked at all.

## What it measures, and what it does not

Each process loops ONE stage back to back and reports its own per-iteration times. Run alone
to get `A_iso` / `B_iso`; run both at once to get `A_co` / `B_co`; `s` and `p` follow.

It is **not** a KV-handoff pipeline: the expert works from its own cache, so there is no
`T_pipe` here and no fresh-KV question — nothing crosses a process boundary, so the stale-KV
hazard non-negotiable 1 guards against does not exist in this mechanism. What it isolates is
exactly the GPU-sharing question, with the host-side coupling removed. The full pipeline's
`T_pipe` still needs the in-process harness, and §4 of the report says why that is blocked.

Two processes may also **time-slice** rather than co-run: two CUDA contexts on one GPU are
not guaranteed to share it spatially. That is itself a result — E11f found two processes left
host segments unaffected at 1.002x — and nsys is what distinguishes the two. Recorded as an
open question here rather than assumed either way.

## Steady-state handshake

Each role warms up, writes a ready-file, then waits for its peer's ready-file before timing.
Without that, one process would still be compiling while the other measured, and the "co-run"
window would contain no co-run. `--peer` names the file to wait for; omitting it means
isolated.

    # isolated
    python bench/corun_2proc.py --role prefix --rung bf16_both --seconds 60 --out P_iso.json
    # co-run (two shells, or the orchestrator below)
    python bench/corun_2proc.py --role prefix --rung bf16_both --seconds 60 \
        --ready /tmp/e12/p.ready --peer /tmp/e12/e.ready --out P_co.json
    python bench/corun_2proc.py --role expert --rung bf16_both --seconds 60 \
        --ready /tmp/e12/e.ready --peer /tmp/e12/p.ready --out E_co.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import corun as CR              # noqa: E402  the stages, shared so both roles run one code path
import phase_builders as PB     # noqa: E402
import pi0_common as P          # noqa: E402


def wait_for(path: Path, timeout: float, label: str) -> bool:
    """Block until `path` exists. Returns False on timeout rather than measuring anyway."""
    t0 = time.perf_counter()
    while not path.exists():
        if time.perf_counter() - t0 > timeout:
            print(f"  TIMEOUT waiting for {label} ({path}) after {timeout:.0f}s -- refusing "
                  f"to time a co-run window in which the peer may not be running", flush=True)
            return False
        time.sleep(0.05)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("prefix", "expert"), required=True)
    ap.add_argument("--rung", default="bf16_both",
                    help="the E7 baseline rung. Usable here because each process has ONE "
                         "sequential call stream, so CUDAGraphTrees never sees two streams.")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--fixtures", default="bench/fixtures/e12")
    ap.add_argument("--ready", default="")
    ap.add_argument("--peer", default="")
    ap.add_argument("--peer-timeout", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nosync", action="store_true")
    ap.add_argument("--sampler", choices=("none", "rails", "rails+tegrastats"),
                    default="none",
                    help="Reproduce E7's measurement conditions, to test whether E7's `ae` "
                         "time carries a sampler artefact. power_core.measure() starts BOTH "
                         "a 20 Hz Rails PYTHON THREAD (which takes the GIL from the kernel "
                         "issue loop) and a tegrastats SUBPROCESS at 100 ms (which competes "
                         "for CPU but not the GIL). They are separable and are separated "
                         "here, because this phase is CPU-issue-bound (measured issue/wall "
                         "= 1.0000) and so is exactly what a GIL thief would slow.")
    args = ap.parse_args()

    import model_registry as MR                                       # noqa: PLC0415
    gate = P.gate()
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    if args.nosync:
        print("  " + CR.patch_pi0_nosync(policy), flush=True)

    rung_note = ""
    if args.rung and args.rung != "eager":
        import opt_ladder as OL                                       # noqa: PLC0415
        compiled, prep, prec, comp_s, fails, _ = OL.apply_rung(policy, spec, args.rung)
        if fails:
            raise SystemExit(f"rung {args.rung} failed to compile: {fails}")
        rung_note = f"{args.rung}: prep=[{prep}] compiled={compiled} matmul={prec}"
        print(f"  {rung_note} in {comp_s:.1f} s", flush=True)

    fixture = json.loads((REPO / args.fixtures / "obs_0.json").read_text())
    batch, _, _ = P.build_batch(policy, fixture, fixtures=REPO / args.fixtures,
                                checkpoint=spec.checkpoint, revision=spec.revision)
    ctx = PB.build_ctx(policy, spec, batch)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda",
                        dtype=torch.float32)

    # One closure per role, so both roles go through bench/corun.py's stage functions and
    # neither gets a private code path.
    if args.role == "prefix":
        def once():
            CR.prefix_stage(ctx)
    else:
        pkv, pad = CR.prefix_stage(ctx)      # the expert works from its OWN cache: nothing
        def once():                          # crosses a process boundary, so there is no
            CR.expert_stage(ctx, pkv, pad, noise, args.steps)   # fresh-KV question here

    for _ in range(args.warm):
        once()
    torch.cuda.synchronize()

    if args.ready:
        Path(args.ready).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ready).write_text(f"{args.role} warm at {time.time()}\n")
    peer_ok = True
    if args.peer:
        print(f"  warm; waiting for peer {args.peer}", flush=True)
        peer_ok = wait_for(Path(args.peer), args.peer_timeout, "peer")

    # Optional interference, started AFTER warm-up and the peer handshake so it covers only
    # the timed window -- the same placement power_core.measure() uses.
    rails = teg = None
    if args.sampler != "none":
        import power_core as PC                                       # noqa: PLC0415
        rails = PC.Rails(20.0)
        rails.start()
        print("  sampler: Rails thread at 20 Hz started", flush=True)
        if args.sampler == "rails+tegrastats":
            import subprocess                                         # noqa: PLC0415
            teg_log = Path(args.out).with_suffix(".tegrastats.log")
            teg_log.parent.mkdir(parents=True, exist_ok=True)
            teg = subprocess.Popen(                                   # noqa: S603
                ["sudo", "-n", "tegrastats", "--interval", "100",
                 "--logfile", str(teg_log)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("  sampler: tegrastats subprocess at 100 ms started", flush=True)

    samples: list[float] = []
    t0 = time.perf_counter()
    end = t0 + args.seconds
    while time.perf_counter() < end:
        a = time.perf_counter()
        once()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - a) * 1e3)
    wall = time.perf_counter() - t0

    if rails is not None:
        rails.stop.set()
        rails.join(timeout=5)
    if teg is not None:
        import power_core as PC                                       # noqa: PLC0415
        PC.stop_tegrastats()          # 22 orphaned tegrastats once corrupted a run
        teg.wait(timeout=10)

    s = CR.summarize(samples)
    out = {"role": args.role, "rung": args.rung, "nosync": args.nosync,
           "sampler": args.sampler,
           "rails_samples": len(rails.samples) if rails is not None else 0,
           "steps": args.steps, "condition": "corun" if args.peer else "isolated",
           "peer_ready_seen": peer_ok, "wall_s": round(wall, 4),
           "iterations": len(samples), "throughput_per_s": round(len(samples) / wall, 4),
           "power_mode": gate["power_mode"], "sm_count": gate["sm_count"], "gate": gate,
           "rung_note": rung_note, "samples_ms": samples, **s}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"  {args.role} {out['condition']}: median {s['median_ms']:.2f} ms  "
          f"n={s['n']}  peer_seen={peer_ok}  -> {args.out}", flush=True)
    return 0 if peer_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
