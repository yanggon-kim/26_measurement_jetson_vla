"""E12 (T014): co-run π0's VLM prefix and action expert on one Orin GPU.

Specification: `docs/E12_corun/README.md`. Binding: protocol §14, §13.7, §1, §3.5, §5.

The question, in one line: prefix(k) on one CUDA stream while expert(k−1) runs on another,
each frame using **its own** KV cache — does the expert slow down, and by how much? The
decisive number is `s = B_co / B_iso` against `r = A_iso / B_iso` (E7: 1.50 at N = 10).
`s < r` means one GPU overlaps the stages essentially ideally and the frame-rate argument
for moving the expert off the GPU is gone; `s > r` means contention is real. The research
side has a stake in one outcome and this harness must not.

## Why the stages are new code rather than `bench/phase_loops.py`

`phase_builders.make_ae` calls `prefix_once` **once**, closes over the resulting cache, and
replays the denoising loop against that same cache on every invocation. For E7's isolated
action-expert loop that is exactly right — it isolates the expert from the prefix. For E12
it is **structurally the stale-KV pattern non-negotiable 1 forbids**: every frame would read
one cache, which is the mechanism (V-AEFusion's, E8's stale-prefix stretch) this experiment
is explicitly *not* measuring. It also reuses one `noise` tensor, so every frame would
denoise the same trajectory.

So `expert_stage` below takes the cache and the noise as **arguments**. `prefix_once` is
reused unchanged — it already returns a fresh `(pkv, pad)` per call.

## The per-step timestep tensor is kept, deliberately

`expert_stage` mirrors `make_ae`'s body including
`torch.tensor(t, dtype=..., device=...)` inside the denoising loop. §7 of the spec flags it
as a candidate host synchronisation, and precomputing the N timesteps once would be an
allowed harness change. It is kept for now because non-negotiable 3 asks the session's
`B_iso` to be reconciled against E7's 167.36 ms, and the cheapest way to make that
comparison meaningful is to run the same code E7 ran. If the diagnostic pass shows this sync
dominates, precomputing becomes a **labelled variant** with its own isolated times and its
own equivalence pass — not a silent change to the baseline.

## Issuing

One Python thread drives both streams (§13.7; T010's two-stream numbers were GIL artefacts).
Per §7 the expert of frame k−1 is enqueued **before** the prefix of frame k, so a cheap
stage never waits behind the issue of an expensive one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import phase_builders as PB      # noqa: E402
import pi0_common as P           # noqa: E402


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def prefix_stage(ctx):
    """vision + VLM prefix for ONE frame -> that frame's own (pkv, pad).

    `phase_builders.prefix_once` already computes a fresh cache per call; it is reused
    unchanged so the prefix executes exactly the code E7 timed.
    """
    return PB.prefix_once(ctx)


def expert_stage(ctx, pkv, pad, noise, num_steps: int):
    """The N denoising steps for ONE frame, against THAT frame's cache and noise.

    The cache and the noise are arguments, not captured state: that is the whole difference
    from `phase_builders.make_ae` and the reason this function exists (see the module
    docstring). Body otherwise identical, including the protocol §2 deviation of a bounded
    Python loop rather than `sample_actions`' GPU-tensor `while`.
    """
    m, spec = ctx["model"], ctx["spec"]
    bsize = pad.shape[0]
    dt = -1.0 / num_steps
    state = ctx["state"]
    has_state_arg = spec.name == "pi0"
    with torch.no_grad():
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            tt = torch.tensor(t, dtype=torch.float32, device=x_t.device).expand(bsize)
            if has_state_arg:
                v_t = m.denoise_step(state, pad, pkv, x_t, tt)
            else:
                v_t = m.denoise_step(prefix_pad_masks=pad, past_key_values=pkv,
                                     x_t=x_t, timestep=tt)
            x_t = x_t + dt * v_t
    return x_t


def _record_stream(obj, stream):
    """Tell the caching allocator that `stream` will read these tensors.

    A tensor produced on the prefix stream and consumed on the expert stream can be freed
    and handed to another allocation while the expert is still reading it, unless the
    consuming stream is recorded against it. That is precisely a corrupted-KV bug, and the
    equivalence check is what would catch it — this is what stops it happening. Walks the
    cache structure because `past_key_values` is a nest of lists/tuples/dicts, not a tensor.
    """
    if torch.is_tensor(obj):
        if obj.is_cuda:
            obj.record_stream(stream)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _record_stream(x, stream)
    elif isinstance(obj, dict):
        for x in obj.values():
            _record_stream(x, stream)
    elif hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):   # transformers Cache
        _record_stream(obj.key_cache, stream)
        _record_stream(obj.value_cache, stream)


# --------------------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------------------
def load_frames(policy, spec, fixtures_dir: Path, n_frames: int, seed0: int):
    """One context + one noise tensor per frame, cycling the E12 observation set.

    §6.1: with a single observation every frame's KV is identical and a stale-KV bug is
    invisible, so the check would have no power. Each frame also gets its own noise seed,
    for the same reason applied to the denoising trajectory.
    """
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    files = [fixtures_dir / f["file"] for f in manifest["frames"]]
    if len(files) < 3:
        raise SystemExit(f"only {len(files)} observations in {fixtures_dir}; §6.1 wants >= 3 "
                         f"or a stale-KV bug is invisible. Run bench/e12_fixtures.py.")
    cfg = policy.config
    ctxs, obs_ids = [], []
    for f in files:
        fixture = json.loads(f.read_text())
        batch, _, _ = P.build_batch(policy, fixture, fixtures=fixtures_dir,
                                    checkpoint=spec.checkpoint, revision=spec.revision)
        ctxs.append(PB.build_ctx(policy, spec, batch))
        obs_ids.append(f.stem)
    frames = []
    for k in range(n_frames):
        i = k % len(ctxs)
        g = torch.Generator(device="cuda").manual_seed(seed0 + k)
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda",
                            dtype=torch.float32, generator=g)
        frames.append({"k": k, "ctx": ctxs[i], "obs_id": obs_ids[i],
                       "seed": seed0 + k, "noise": noise})
    return frames, manifest


# --------------------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------------------
def run_serial(frames, num_steps: int, keep_actions: bool = False):
    """Both stages back to back on one stream: the T_serial condition and the reference."""
    out, samples = [], []
    for fr in frames:
        torch.cuda.synchronize()
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record()
        pkv, pad = prefix_stage(fr["ctx"])
        e1.record()
        act = expert_stage(fr["ctx"], pkv, pad, fr["noise"], num_steps)
        e2.record()
        torch.cuda.synchronize()
        samples.append({"frame": fr["k"], "obs_id": fr["obs_id"], "seed": fr["seed"],
                        "prefix_ms": e0.elapsed_time(e1), "expert_ms": e1.elapsed_time(e2),
                        "frame_ms": e0.elapsed_time(e2)})
        out.append(act.detach().clone() if keep_actions else None)
    return out, samples


def run_isolated(frames, num_steps: int, stage: str, stream):
    """One stage alone, through the SAME harness and stream the co-run uses (§3 def).

    The other stream is empty. For the expert this still runs a prefix first, because the
    expert needs a cache — but only the expert's own events are timed, which is what
    `B_iso` means.
    """
    samples = []
    for fr in frames:
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            if stage == "prefix":
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                prefix_stage(fr["ctx"])
                e1.record(stream)
            else:
                pkv, pad = prefix_stage(fr["ctx"])      # untimed: the expert needs a cache
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                expert_stage(fr["ctx"], pkv, pad, fr["noise"], num_steps)
                e1.record(stream)
        torch.cuda.synchronize()
        samples.append({"frame": fr["k"], "obs_id": fr["obs_id"], "seed": fr["seed"],
                        "ms": e0.elapsed_time(e1)})
    return samples


def run_pipeline(frames, num_steps: int, stream_p, stream_e, keep_actions: bool = False,
                 stale: bool = False):
    """The steady-state pipeline: prefix(k) on stream_p, expert(k-1) on stream_e.

    Free-running, gated only by cross-stream events — not lockstep with a per-frame
    barrier, which would serialise the stages and understate the overlap this experiment
    exists to measure.

    `stale=True` is the NEGATIVE CONTROL of §6.3: the expert is handed the WRONG frame's
    cache. It must fail the equivalence check. If it does not, the check has no power and
    nothing this harness reports is verified.
    """
    pending: dict[int, dict] = {}
    actions: dict[int, torch.Tensor] = {}
    samples: list[dict] = []
    stale_src: dict | None = None        # the cache from TWO frames back (negative control)

    for fr in frames:
        k = fr["k"]
        # (1) expert of the PREVIOUS frame first, per §7: a cheap stage must never wait
        #     behind the issue of an expensive one.
        prev = pending.pop(k - 1, None)
        if prev is not None:
            # NEGATIVE CONTROL (§6.3): hand expert(k-1) the cache of frame k-2 instead of
            # its own. The first version of this reached for `pending[k]` -- frame k's
            # cache -- but at this point frame k's prefix has not been issued yet (the
            # expert goes first, per §7), so it fell back to the CORRECT cache. The control
            # would have been identical to the real path, passed the check, and been
            # reported as "the check has no power" when the control was simply never stale.
            # Two frames back is available, is genuinely the wrong frame, and is what §6.3
            # offers as the alternative ("or the previous frame's").
            use_pkv, use_pad = prev["pkv"], prev["pad"]
            if stale and stale_src is not None:
                use_pkv, use_pad = stale_src["pkv"], stale_src["pad"]
            stream_e.wait_event(prev["ev_done"])  # gate: prefix(k-1) has produced the cache
            with torch.cuda.stream(stream_e):
                # start AFTER the wait, so waiting for the other stage is not counted as
                # work (§3, "Stage time under co-run").
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream_e)
                act = expert_stage(prev["ctx"], use_pkv, use_pad, prev["noise"], num_steps)
                e1.record(stream_e)
            prev.update(ev_e0=e0, ev_e1=e1, act=act,
                        used_stale=bool(stale and stale_src is not None))
            stale_src = prev                      # frame k-1 becomes "two back" next time

        # (2) prefix of THIS frame
        with torch.cuda.stream(stream_p):
            p0 = torch.cuda.Event(enable_timing=True)
            p1 = torch.cuda.Event(enable_timing=True)
            p0.record(stream_p)
            pkv, pad = prefix_stage(fr["ctx"])
            p1.record(stream_p)
            done = torch.cuda.Event()
            done.record(stream_p)
        # The expert stream will read this cache: tell the allocator, and hold the
        # reference in `pending` until that expert has been issued.
        _record_stream(pkv, stream_e)
        _record_stream(pad, stream_e)
        pending[k] = {"k": k, "ctx": fr["ctx"], "obs_id": fr["obs_id"], "seed": fr["seed"],
                      "noise": fr["noise"], "pkv": pkv, "pad": pad,
                      "ev_done": done, "ev_p0": p0, "ev_p1": p1}

        if prev is not None:
            samples.append(prev)

    # drain the last expert
    last = pending.pop(max(pending) if pending else -1, None)
    if last is not None:
        use_pkv, use_pad = last["pkv"], last["pad"]
        if stale and stale_src is not None:       # same control as the loop above
            use_pkv, use_pad = stale_src["pkv"], stale_src["pad"]
        stream_e.wait_event(last["ev_done"])
        with torch.cuda.stream(stream_e):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(stream_e)
            act = expert_stage(last["ctx"], use_pkv, use_pad, last["noise"], num_steps)
            e1.record(stream_e)
        last.update(ev_e0=e0, ev_e1=e1, act=act,
                    used_stale=bool(stale and stale_src is not None))
        samples.append(last)

    torch.cuda.synchronize()
    rows = []
    for r in samples:
        if keep_actions and r.get("act") is not None:
            actions[r["k"]] = r["act"].detach().clone()
        rows.append({"frame": r["k"], "obs_id": r["obs_id"], "seed": r["seed"],
                     # Whether the negative control actually substituted a wrong cache for
                     # THIS frame. Frame 0 cannot be stale -- there is no earlier frame --
                     # so scoring it as "the control failed to change the actions" would
                     # report EQUIV_FAIL on a perfectly correct run.
                     "used_stale": bool(r.get("used_stale", False)),
                     "prefix_ms": r["ev_p0"].elapsed_time(r["ev_p1"]),
                     "expert_ms": (r["ev_e0"].elapsed_time(r["ev_e1"])
                                   if "ev_e0" in r else float("nan"))})
    return actions, rows


# --------------------------------------------------------------------------------------
# equivalence (§6) — the gate on every number this harness reports
# --------------------------------------------------------------------------------------
def equivalence(frames, num_steps: int, stream_p, stream_e, path: str) -> list[dict]:
    """Pipelined actions vs serial actions, frame by frame, plus the negative control.

    §6 in three parts, all of which have to hold:
      1. distinct observation AND distinct noise seed per frame (built in `load_frames`);
      2. pipelined == serial for the same frame — in eager, nothing but the schedule
         differs, so bitwise identity is the expectation;
      3. a deliberately stale variant that MUST fail. Without it the check has no power:
         a comparison that cannot tell a corrupted cache from a correct one proves nothing.
    """
    ref, _ = run_serial(frames, num_steps, keep_actions=True)
    got, _ = run_pipeline(frames, num_steps, stream_p, stream_e, keep_actions=True)
    bad, bad_rows = run_pipeline(frames, num_steps, stream_p, stream_e, keep_actions=True,
                                 stale=True)
    # Frames where the control actually substituted a wrong cache. Frame 0 never can, so
    # judging it would turn a correct run into EQUIV_FAIL.
    staled = {r["frame"] for r in bad_rows if r.get("used_stale")}

    rows = []
    for fr in frames:
        k = fr["k"]
        r = ref[k]
        for mech, actions in (("pipeline", got), ("stale_control", bad)):
            a = actions.get(k)
            if r is None or a is None:
                continue
            if mech == "stale_control" and k not in staled:
                continue        # no wrong cache was available for this frame
            d = (r - a).abs()
            rel = (d / r.abs().clamp_min(1e-12)).max().item()
            rows.append({"mechanism": mech, "path": path, "frame": k,
                         "obs_id": fr["obs_id"], "seed": fr["seed"],
                         "max_abs_diff": d.max().item(), "max_rel_diff": rel,
                         "bitwise_identical": bool(torch.equal(r, a)),
                         # A stale cache must CHANGE the actions; the pipeline must not.
                         "verdict": ("PASS" if (torch.equal(r, a) if mech == "pipeline"
                                                else not torch.equal(r, a)) else "FAIL")})
    return rows


def frame_to_frame_scale(frames, num_steps: int) -> float:
    """How different the serial actions are between consecutive frames.

    §6.1 asks for this explicitly: it is the scale at which a stale-KV mix-up would show
    up. If it were near zero the observations would not be materially different and the
    equivalence check would be weak however it is written.
    """
    acts, _ = run_serial(frames[:min(4, len(frames))], num_steps, keep_actions=True)
    diffs = [(acts[i] - acts[i - 1]).abs().max().item()
             for i in range(1, len(acts)) if acts[i] is not None and acts[i - 1] is not None]
    return max(diffs) if diffs else 0.0


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("equivalence",), default="equivalence")
    ap.add_argument("--path", choices=("eager", "compiled", "both"), default="both",
                    help="§6.2 wants BOTH: eager, where nothing but the schedule differs, "
                         "and the compiled bf16_both path that is actually timed, where "
                         "static CUDA-graph buffers and cross-stream reuse are hazards "
                         "only that path has")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fixtures", default="bench/fixtures/e12")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    import model_registry as MR                                       # noqa: PLC0415
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    spec = MR.get("pi0")
    fixtures = REPO / args.fixtures

    paths = ("eager", "compiled") if args.path == "both" else (args.path,)
    rows: list[dict] = []
    notes: list[str] = []
    scale = None

    for path in paths:
        # A fresh policy per path: applying the rung mutates the module in place, so an
        # eager check after a compiled one would not be eager.
        policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
        # assert_real_weights RAISES SystemExit on a mismatch -- calling it IS the check.
        # The first version of this line wrapped it in `if not real.get("ok", True)`, which
        # is dead code: the returned dict has no "ok" key, so the default made the
        # conditional always false. A guard that cannot fire is worse than none, because it
        # reads as protection. Called BEFORE apply_rung, because bf16_both casts 74
        # layernorms and the vision tower and E7 saw the check then report "325/400 tensors
        # match" -- a true statement about the cast, not about the checkpoint.
        real = P.assert_real_weights(policy, spec=spec)
        rung_note = ""
        if path == "compiled":
            import opt_ladder as OL                                   # noqa: PLC0415
            compiled, prep_note, prec, comp_s, fails, _ = OL.apply_rung(
                policy, spec, "bf16_both")
            if fails:
                raise SystemExit(f"bf16_both failed to compile: {fails}")
            rung_note = f"bf16_both: prep=[{prep_note}] compiled={compiled} matmul={prec}"
            print(f"  rung {rung_note} in {comp_s:.1f} s", flush=True)

        frames, manifest = load_frames(policy, spec, fixtures, args.frames, args.seed)
        stream_p = torch.cuda.Stream()
        stream_e = torch.cuda.Stream()

        # Warm up both paths: a first-call compile or autotune inside the measured region
        # would look like anything at all.
        run_serial(frames[:2], args.steps)
        torch.cuda.synchronize()

        if scale is None:
            scale = frame_to_frame_scale(frames, args.steps)
            print(f"  frame-to-frame action difference (serial): {scale:.6e}", flush=True)
            notes.append(
                f"Consecutive serial frames differ by max|d|={scale:.6e} in the action "
                f"chunk. §6.1 asks for this: it is the scale at which a stale-KV mix-up "
                f"would show, and a near-zero value would mean the observations are not "
                f"materially different and the check is weak however it is written.")

        r = equivalence(frames, args.steps, stream_p, stream_e, path)
        rows += r
        pipe = [x for x in r if x["mechanism"] == "pipeline"]
        ctrl = [x for x in r if x["mechanism"] == "stale_control"]
        print(f"  [{path}] pipeline  : {sum(x['verdict']=='PASS' for x in pipe)}/{len(pipe)} "
              f"PASS  max_abs_diff={max((x['max_abs_diff'] for x in pipe), default=0):.3e}")
        print(f"  [{path}] stale ctrl: {sum(x['verdict']=='PASS' for x in ctrl)}/{len(ctrl)} "
              f"PASS (PASS here means it CHANGED the actions, as it must)  "
              f"max_abs_diff={max((x['max_abs_diff'] for x in ctrl), default=0):.3e}",
              flush=True)
        if rung_note:
            notes.append(rung_note)
        del policy
        torch.cuda.empty_cache()

    # ---- verdict
    n_fail = sum(x["verdict"] == "FAIL" for x in rows)
    ctrl_rows = [x for x in rows if x["mechanism"] == "stale_control"]
    ctrl_detects = sum(x["verdict"] == "PASS" for x in ctrl_rows)
    ok = n_fail == 0 and ctrl_detects > 0
    print()
    print(f"  VERDICT: {'PASS' if ok else 'EQUIV_FAIL'}  "
          f"({n_fail} failing row(s); negative control detected "
          f"{ctrl_detects}/{len(ctrl_rows)})", flush=True)
    if ctrl_detects == 0:
        print("  *** NEGATIVE CONTROL DID NOT FAIL -- the check has no power and no E12 "
              "number is verified (§6.3) ***", flush=True)

    if args.outdir:
        out = REPO / args.outdir
        (out / "raw").mkdir(parents=True, exist_ok=True)
        import csv                                                    # noqa: PLC0415
        cols = ["mechanism", "path", "frame", "obs_id", "seed", "max_abs_diff",
                "max_rel_diff", "bitwise_identical", "verdict"]
        with (out / "equivalence.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        (out / "equivalence.md").write_text(
            "# E12 fresh-KV equivalence (`docs/E12_corun/README.md` §6)\n\n"
            f"**VERDICT: {'PASS' if ok else 'EQUIV_FAIL'}** — {n_fail} failing row(s); the "
            f"negative control changed the actions in {ctrl_detects}/{len(ctrl_rows)} rows.\n\n"
            "Non-negotiable 1: each frame's expert must read that frame's own KV cache. "
            "A pipeline that shares, reuses or overwrites a cache across frames is a "
            "different mechanism (stale KV) and would invalidate every timing.\n\n"
            "`stale_control` is the §6.3 negative control: expert(k−1) is handed frame "
            "k−2's cache. **PASS for that row means it CHANGED the actions, as it must** — "
            "if a deliberately corrupted cache were indistinguishable from a correct one, "
            "the check would have no power.\n\n"
            + "\n".join(f"* {n}" for n in notes) + "\n\n"
            "| mechanism | path | frames | max abs diff | bitwise identical | verdict |\n"
            "|---|---|---:|---:|---|---|\n"
            + "\n".join(
                f"| `{m}` | {p} | {len([x for x in rows if x['mechanism']==m and x['path']==p])} "
                f"| {max((x['max_abs_diff'] for x in rows if x['mechanism']==m and x['path']==p), default=0):.3e} "
                f"| {all(x['bitwise_identical'] for x in rows if x['mechanism']==m and x['path']==p)} "
                f"| {'PASS' if all(x['verdict']=='PASS' for x in rows if x['mechanism']==m and x['path']==p) else 'FAIL'} |"
                for m in ("pipeline", "stale_control") for p in paths) + "\n")
        print(f"  wrote {out}/equivalence.csv and equivalence.md", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
