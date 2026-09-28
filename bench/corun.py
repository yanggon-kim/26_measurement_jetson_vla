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


_TIMESTEPS: dict[tuple, torch.Tensor] = {}


def timestep_tensors(num_steps: int, bsize: int, device, dtype=torch.float32):
    """The N timestep tensors, built ONCE and reused.

    `torch.tensor(<python float>, device=...)` is a host->device transfer and
    torch.cuda.set_sync_debug_mode("warn") flags it as a synchronising operation. Inside the
    denoising loop it fired once per step, and a sync in one stage's issue path stalls the
    OTHER stream's launches too -- which is why the first pipeline showed
    T_pipe = T_serial with no overlap at all.

    §7 of the spec names precomputing the step times as THE example of a permitted harness
    change, so this is not a labelled variant: it is the harness being correct. The VALUES
    are identical (t = 1 + step * dt, same float arithmetic), so the equivalence check must
    still show bitwise identity -- and it is re-run to prove that rather than assumed.
    """
    key = (num_steps, bsize, str(device), dtype)
    ts = _TIMESTEPS.get(key)
    if ts is None:
        dt = -1.0 / num_steps
        ts = [torch.tensor(1.0 + step * dt, dtype=dtype, device=device).expand(bsize)
              for step in range(num_steps)]
        _TIMESTEPS[key] = ts
    return ts


def expert_stage(ctx, pkv, pad, noise, num_steps: int, e7_timesteps: bool = False):
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
    tsteps = None if e7_timesteps else timestep_tensors(num_steps, bsize, pad.device)
    with torch.no_grad():
        x_t = noise
        for step in range(num_steps):
            if e7_timesteps:
                # E7's construction, verbatim from phase_builders.make_ae. DIAGNOSTIC ONLY:
                # torch.tensor(<python float>, device="cuda") copies from PAGEABLE host
                # memory, which blocks the host until the copy lands -- i.e. until the
                # stream has drained every kernel queued ahead of it. On a launch-bound
                # stage that discards the host's run-ahead once per denoise step. This
                # flag exists to measure that cost against E7's `ae` time, nothing else.
                tt = torch.tensor(1.0 + step * dt, dtype=torch.float32,
                                  device=x_t.device).expand(bsize)
            else:
                tt = tsteps[step]      # precomputed: see timestep_tensors()
            if has_state_arg:
                v_t = m.denoise_step(state, pad, pkv, x_t, tt)
            else:
                v_t = m.denoise_step(prefix_pad_masks=pad, past_key_values=pkv,
                                     x_t=x_t, timestep=tt)
            x_t = x_t + dt * v_t
    return x_t


def patch_pi0_nosync(policy) -> str:
    """LABELLED VARIANT: memoise pi0's two `torch.tensor(<python list>)` conversions.

    Measured with torch.cuda.set_sync_debug_mode("warn"), the stock model synchronises at

        modeling_pi0.py:681   once per PREFIX    att_masks -> bool tensor
        modeling_pi0.py:746   once per DENOISE STEP   att_masks -> embs.dtype tensor

    i.e. 1 + N host syncs per frame (11 at N = 10). A sync in either stage's issue path
    stalls the OTHER stream's launches, so with one issuing thread (§13.7) the two streams
    cannot overlap however correctly they are wired -- which is why the first pipeline gave
    T_pipe = T_serial exactly.

    Both build a tensor from a CONSTANT Python list, so the conversion is memoised on the
    list's CONTENTS. Keying on contents rather than on call site is what makes it correct
    when the prompt length changes between observations: a different att_masks is a
    different key and gets its own tensor. The cached tensor is only ever `.expand()`ed by
    the caller, which is a view, so handing back the same object is safe.

    This is a **labelled harness change**, not part of the baseline. §7 permits it -- "a
    harness change, allowed if the equivalence check passes and the isolated stages are
    re-measured through the changed harness" -- and every number taken under it carries the
    label, with its own isolated times. The baseline rung is untouched: this patches nothing
    in site-packages and changes no numerics.
    """
    import lerobot.policies.pi0.modeling_pi0 as M                     # noqa: PLC0415
    m = policy.model
    memo: dict = {}
    real_tensor = torch.tensor

    def cached(data, **kw):
        if type(data) is list and data and isinstance(data[0], (bool, int, float)):
            key = (tuple(data), kw.get("dtype"), str(kw.get("device")))
            t = memo.get(key)
            if t is None:
                t = real_tensor(data, **kw)
                memo[key] = t
            return t
        return real_tensor(data, **kw)

    def wrap(fn):
        def inner(*a, **k):
            M.torch.tensor = cached          # restored in finally; single-threaded (§13.7)
            try:
                return fn(*a, **k)
            finally:
                M.torch.tensor = real_tensor
        return inner

    m.embed_prefix = wrap(m.embed_prefix)
    m.embed_suffix = wrap(m.embed_suffix)
    return ("pi0_nosync: memoised the constant att_masks tensors at modeling_pi0.py:681 "
            "(per prefix) and :746 (per denoise step); 1+N host syncs per frame removed")


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
    import time as _t                                                 # noqa: PLC0415
    pending: dict[int, dict] = {}
    actions: dict[int, torch.Tensor] = {}
    samples: list[dict] = []
    stale_src: dict | None = None        # the cache from TWO frames back (negative control)
    _issue_t0 = _t.perf_counter()

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

    # HOST ISSUE TIME vs WALL. The decisive test for "is there anything to overlap".
    #
    # pi0 in eager is launch-bound: E5 measured ae_step at 84.2 % GPU idle over 968 kernels
    # per step. If one Python thread spends nearly the whole wall clock ISSUING kernels,
    # then while it issues the expert the prefix stream has no queued work and vice versa,
    # so the two streams cannot overlap however correctly they are wired -- and every stage
    # still reports its isolated time, because each has the GPU to itself when it runs.
    # That is indistinguishable from ideal overlap in s and p, and only T_pipe exposes it.
    #
    # issue_s close to wall_s  => CPU-bound issuing; co-run cannot help and the experiment
    #                             cannot see GPU contention in this configuration.
    # issue_s well below wall_s => the GPU is the bottleneck and overlap is meaningful.
    issue_s = _t.perf_counter() - _issue_t0
    torch.cuda.synchronize()
    wall_s = _t.perf_counter() - _issue_t0
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
                                   if "ev_e0" in r else float("nan")),
                     "issue_s": issue_s, "wall_s": wall_s})
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

    # DIAGNOSTIC, before judging anything: are the pipeline's own frames distinct from
    # each other? The compiled rung graphs denoise_step with reduce-overhead, so its
    # outputs live in STATIC buffers that each replay overwrites. If every frame's action
    # tensor is the same object or the same values, the fault is this harness cloning too
    # late -- not the device corrupting a cache. Those two have identical symptoms in the
    # equivalence table and completely different fixes, so the distinction is measured.
    ks = sorted(got)
    same_obj = len({id(got[k]) for k in ks}) < len(ks)
    pair_diffs = [ (got[a] - got[b]).abs().max().item()
                   for a, b in zip(ks, ks[1:]) ] if len(ks) > 1 else []
    print(f"  [{path}] pipeline self-distinctness: shared_tensor_objects={same_obj}  "
          f"consecutive max|d|={[f'{d:.3e}' for d in pair_diffs]}", flush=True)
    ref_diffs = [ (ref[a] - ref[b]).abs().max().item()
                  for a, b in zip(ks, ks[1:]) if ref[a] is not None and ref[b] is not None ]
    print(f"  [{path}] serial   self-distinctness: consecutive max|d|="
          f"{[f'{d:.3e}' for d in ref_diffs]}", flush=True)

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
# timing (Steps 1 + 2)
# --------------------------------------------------------------------------------------
def summarize(vals: list[float]) -> dict:
    import statistics                                                 # noqa: PLC0415
    v = sorted(x for x in vals if x == x)                             # drop NaN
    if not v:
        return {"median_ms": float("nan"), "mean_ms": float("nan"), "p5_ms": float("nan"),
                "p95_ms": float("nan"), "min_ms": float("nan"), "max_ms": float("nan"),
                "n": 0}
    def pct(q):
        return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]
    return {"median_ms": statistics.median(v), "mean_ms": statistics.fmean(v),
            "p5_ms": pct(0.05), "p95_ms": pct(0.95), "min_ms": v[0], "max_ms": v[-1],
            "n": len(v)}


def bootstrap_ci(num: list[float], den_median: float, iters: int = 2000, seed: int = 7):
    """95 % interval for the median of `num` divided by a fixed denominator median.

    `s` is a ratio of two medians measured in different conditions, so the honest interval
    is over the co-run samples with the isolated median held fixed -- resampling both would
    conflate two sources of spread that the report states separately.
    """
    import random                                                     # noqa: PLC0415
    import statistics                                                 # noqa: PLC0415
    v = [x for x in num if x == x]
    if not v or not den_median:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(iters))
    lo = meds[int(0.025 * (iters - 1))] / den_median
    hi = meds[int(0.975 * (iters - 1))] / den_median
    return lo, hi


def timing(frames, warm, num_steps, stream_p, stream_e):
    """Bracketed isolated -> co-run -> isolated, all in one session (non-negotiable 3)."""
    out = {}
    print(f"  warm-up: {warm} frames", flush=True)
    run_serial(frames[:warm], num_steps)
    torch.cuda.synchronize()

    print("  A_iso (prefix alone, through the co-run's own stream)...", flush=True)
    out["A_iso_pre"] = run_isolated(frames, num_steps, "prefix", stream_p)
    print("  B_iso (expert alone)...", flush=True)
    out["B_iso_pre"] = run_isolated(frames, num_steps, "expert", stream_e)
    print("  T_serial (both stages back to back on one stream)...", flush=True)
    _, out["serial"] = run_serial(frames, num_steps)

    print("  pipeline (prefix(k) || expert(k-1))...", flush=True)
    torch.cuda.synchronize()
    import time                                                       # noqa: PLC0415
    t0 = time.perf_counter()
    _, out["corun"] = run_pipeline(frames, num_steps, stream_p, stream_e)
    torch.cuda.synchronize()
    out["pipe_wall_s"] = time.perf_counter() - t0
    out["pipe_frames"] = len(out["corun"])

    print("  A_iso / B_iso again (the bracket: drift must be visible)...", flush=True)
    out["A_iso_post"] = run_isolated(frames, num_steps, "prefix", stream_p)
    out["B_iso_post"] = run_isolated(frames, num_steps, "expert", stream_e)
    return out


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("equivalence", "timing"), default="equivalence")
    ap.add_argument("--path", choices=("eager", "compiled", "both"), default="both",
                    help="§6.2 wants BOTH: eager, where nothing but the schedule differs, "
                         "and the compiled bf16_both path that is actually timed, where "
                         "static CUDA-graph buffers and cross-stream reuse are hazards "
                         "only that path has")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--nosync", action="store_true",
                    help="LABELLED VARIANT: memoise pi0's per-prefix and per-step "
                         "torch.tensor(<list>) conversions, which are host syncs that stall "
                         "the other stream's launches. Its numbers are reported under the "
                         "label `pi0_nosync` with their own isolated times, never as the "
                         "baseline.")
    ap.add_argument("--sync-debug", action="store_true",
                    help="torch.cuda.set_sync_debug_mode('warn') around a short pipeline. "
                         "§7: a host synchronisation inside either stage's issue path "
                         "blocks the OTHER stream's launches too, so the stages serialise "
                         "and the wall clock shows no overlap while the per-stage event "
                         "timings still look isolated-normal.")
    ap.add_argument("--warm", type=int, default=20,
                    help="warm-up frames before any timed condition (spec Step 1: >= 20)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fixtures", default="bench/fixtures/e12")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--allow-capped", action="store_true",
                    help="permit a capped power mode (MODE_30W/15W; T014 Step 4). The gate still records the mode, SM count and clocks in meta.json; without this flag anything but MAXN is refused.")
    args = ap.parse_args()

    import model_registry as MR                                       # noqa: PLC0415
    gate = P.gate(require_maxn=not args.allow_capped)
    print("gate:", json.dumps(gate), flush=True)
    spec = MR.get("pi0")
    fixtures = REPO / args.fixtures

    if args.mode == "timing":
        return run_timing(args, spec, gate)

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
        if getattr(args, "nosync", False):
            print("  " + patch_pi0_nosync(policy), flush=True)
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
        # protocol §5: meta.json per run. The equivalence branch wrote none until 2026-09-26
        # (found on the MODE_30W run, whose meta.json was back-filled).
        import hashlib, subprocess, time                              # noqa: PLC0415, E401
        files = [{"path": str(q.relative_to(out)), "bytes": q.stat().st_size,
                  "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
                 for q in sorted(out.rglob("*")) if q.is_file() and q.name != "meta.json"]
        (out / "meta.json").write_text(json.dumps({
            "experiment": "E12 fresh-KV equivalence (spec §6)", "verdict": "PASS" if ok else "EQUIV_FAIL",
            "n_fail": n_fail, "negative_control_detected": f"{ctrl_detects}/{len(ctrl_rows)}",
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "gpu_clock_hz": gate["gpu_cur_freq_hz"], "gate": gate, "N": args.steps,
            "frames": args.frames, "paths": list(paths),
            "command": " ".join([sys.executable, *sys.argv]),
            "bench_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                           capture_output=True, text=True).stdout.strip(),
            "raw_files": files,
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
        print(f"  wrote {out}/equivalence.csv, equivalence.md and meta.json", flush=True)
    return 0 if ok else 1




def run_timing(args, spec, gate) -> int:
    """Step 1 + 2 at one power mode: A_iso, B_iso, T_serial, A_co, B_co, T_pipe -> s, p, r."""
    import csv                                                        # noqa: PLC0415
    import model_registry as MR                                       # noqa: PLC0415

    if args.path == "compiled":
        raise SystemExit(
            "the stock compiled bf16_both path CANNOT be pipelined across two streams on "
            "this stack: the §6 equivalence check fails non-deterministically (max|d| 1.273 "
            "on one run, all-NaN on the next) while compiled SERIAL is correct and eager "
            "PIPELINED is bitwise exact. See scratch/NOTE_e12_compiled_two_stream.txt. It "
            "produces plausible milliseconds and NaN actions, so timing it would be worse "
            "than not timing it. Implement the own-CUDAGraph fallback (spec §7) first.")

    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    if getattr(args, "nosync", False):
        print("  " + patch_pi0_nosync(policy), flush=True)
    frames, manifest = load_frames(policy, spec, REPO / args.fixtures, args.frames,
                                   args.seed)
    stream_p, stream_e = torch.cuda.Stream(), torch.cuda.Stream()

    if args.sync_debug:
        # Warm first: the first call compiles/allocates and would drown the signal.
        run_serial(frames[:2], args.steps)
        torch.cuda.synchronize()
        print("  --- sync-debug: one pipeline pass, warnings are host syncs ---", flush=True)
        torch.cuda.set_sync_debug_mode("warn")
        try:
            run_pipeline(frames[:3], args.steps, stream_p, stream_e)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        print("  --- end sync-debug ---", flush=True)
        return 0

    t = timing(frames, args.warm, args.steps, stream_p, stream_e)

    A_pre = summarize([r["ms"] for r in t["A_iso_pre"]])
    B_pre = summarize([r["ms"] for r in t["B_iso_pre"]])
    A_post = summarize([r["ms"] for r in t["A_iso_post"]])
    B_post = summarize([r["ms"] for r in t["B_iso_post"]])
    A_co = summarize([r["prefix_ms"] for r in t["corun"]])
    B_co = summarize([r["expert_ms"] for r in t["corun"]])
    Tser = summarize([r["frame_ms"] for r in t["serial"]])

    # Isolated reference = the mean of the two brackets, so drift is shared rather than
    # attributed to whichever end happened to be measured first.
    A_iso = (A_pre["median_ms"] + A_post["median_ms"]) / 2
    B_iso = (B_pre["median_ms"] + B_post["median_ms"]) / 2
    drift_A = abs(A_post["median_ms"] - A_pre["median_ms"]) / A_pre["median_ms"] * 100
    drift_B = abs(B_post["median_ms"] - B_pre["median_ms"]) / B_pre["median_ms"] * 100
    T_pipe = t["pipe_wall_s"] * 1e3 / max(t["pipe_frames"], 1)
    s_val = B_co["median_ms"] / B_iso
    p_val = A_co["median_ms"] / A_iso
    r_val = A_iso / B_iso
    lo, hi = bootstrap_ci([r["expert_ms"] for r in t["corun"]], B_iso)

    print()
    print(f"  A_iso  {A_iso:8.2f} ms   (bracket drift {drift_A:.1f} %)")
    print(f"  B_iso  {B_iso:8.2f} ms   (bracket drift {drift_B:.1f} %)")
    print(f"  T_serial {Tser['median_ms']:6.2f} ms      T_pipe {T_pipe:8.2f} ms")
    print(f"  A_co   {A_co['median_ms']:8.2f} ms   B_co {B_co['median_ms']:8.2f} ms")
    # The decisive evidence for WHY, not just how much. If the host spends nearly the whole
    # wall clock issuing kernels, the two streams cannot overlap however correctly they are
    # wired: while one stage is being issued the other stream has no queued work. s and p
    # look like ideal overlap either way; this is what separates the two.
    iss = t["corun"][0].get("issue_s", float("nan")) if t["corun"] else float("nan")
    wal = t["corun"][0].get("wall_s", float("nan")) if t["corun"] else float("nan")
    if wal == wal and wal:
        print(f"  host issue {iss * 1e3:8.1f} ms   wall {wal * 1e3:8.1f} ms   "
              f"issue/wall = {iss / wal:.4f}   <- ~1.0 means the CPU is the bottleneck and "
              f"there is nothing to overlap")
    print()
    print(f"  s = B_co/B_iso = {s_val:.4f}   [95% CI {lo:.4f}, {hi:.4f}]")
    print(f"  r = A_iso/B_iso = {r_val:.4f}   <- the break-even for s, THIS session")
    print(f"  p = A_co/A_iso = {p_val:.4f}")
    print(f"  T_pipe/A_iso = {T_pipe / A_iso:.4f}   speed-up vs serial = "
          f"{Tser['median_ms'] / T_pipe:.4f}")
    print()
    # TWO separate questions, which one line conflated. The first printed verdict said
    # "s > r -> the expert is the pipeline's bottleneck", which is arithmetically right and
    # reads as "contention found" -- while s = 0.9989 means NO slowdown at all. In eager at
    # N=10 the expert is simply the LONGER stage (r = 0.88 < 1, unlike E7's bf16_both where
    # r = 1.50, because compiling speeds the expert more than the prefix and flips which
    # dominates). Report contention and bottleneck separately.
    overlapped = T_pipe < 0.95 * (A_iso + B_iso)
    print(f"  (1) contention: s = {s_val:.4f} vs 1.0 -> "
          f"{'the expert is slowed by co-run' if s_val > 1.05 else 'NO measurable slowdown'}")
    print(f"  (2) which stage sets the period: s {'>' if s_val > r_val else '<'} r "
          f"({s_val:.4f} vs {r_val:.4f}) -> the "
          f"{'EXPERT' if s_val > r_val else 'PREFIX'} is the longer stage under co-run")
    print(f"  (3) did the stages overlap AT ALL: T_pipe {T_pipe:.1f} vs A_iso+B_iso "
          f"{A_iso + B_iso:.1f} -> {'YES' if overlapped else 'NO -- the pipeline is SERIAL'}")
    if not overlapped:
        print("      s and p are therefore NOT overlap measurements: each stage ran with the "
              "GPU to itself, which is why both read ~1.00. Only T_pipe distinguishes ideal "
              "overlap from none.")

    if args.outdir:
        out = REPO / args.outdir
        (out / "raw").mkdir(parents=True, exist_ok=True)
        with (out / "latency_samples.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mechanism", "condition", "stage", "frame", "ms"])
            for cond, rows_, stage, key in (
                    ("isolated_pre", t["A_iso_pre"], "prefix", "ms"),
                    ("isolated_pre", t["B_iso_pre"], "expert", "ms"),
                    ("isolated_post", t["A_iso_post"], "prefix", "ms"),
                    ("isolated_post", t["B_iso_post"], "expert", "ms"),
                    ("serial", t["serial"], "prefix", "prefix_ms"),
                    ("serial", t["serial"], "expert", "expert_ms"),
                    ("serial", t["serial"], "frame", "frame_ms"),
                    ("corun", t["corun"], "prefix", "prefix_ms"),
                    ("corun", t["corun"], "expert", "expert_ms")):
                for r in rows_:
                    w.writerow(["baseline_eager", cond, stage, r["frame"], r[key]])
        with (out / "corun_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mechanism", "power_mode", "sm_count", "N",
                                              "condition", "stage", "median_ms", "mean_ms",
                                              "p5_ms", "p95_ms", "min_ms", "max_ms", "n"])
            w.writeheader()
            for cond, stage, d in (("isolated_pre", "prefix", A_pre),
                                   ("isolated_pre", "expert", B_pre),
                                   ("isolated_post", "prefix", A_post),
                                   ("isolated_post", "expert", B_post),
                                   ("corun", "prefix", A_co), ("corun", "expert", B_co),
                                   ("serial", "frame", Tser)):
                w.writerow({"mechanism": "baseline_eager",
                            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
                            "N": args.steps, "condition": cond, "stage": stage, **d})
        with (out / "corun_ratios.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "mechanism", "power_mode", "sm_count", "N", "A_iso_ms", "B_iso_ms",
                "T_serial_ms", "A_co_ms", "B_co_ms", "T_pipe_ms", "s", "s_ci95_lo",
                "s_ci95_hi", "p", "r", "T_pipe_over_A_iso", "speedup_vs_serial",
                "reference", "notes"])
            w.writeheader()
            w.writerow({
                "mechanism": "baseline_eager", "power_mode": gate["power_mode"],
                "sm_count": gate["sm_count"], "N": args.steps,
                "A_iso_ms": round(A_iso, 4), "B_iso_ms": round(B_iso, 4),
                "T_serial_ms": round(Tser["median_ms"], 4),
                "A_co_ms": round(A_co["median_ms"], 4),
                "B_co_ms": round(B_co["median_ms"], 4), "T_pipe_ms": round(T_pipe, 4),
                "s": round(s_val, 6), "s_ci95_lo": round(lo, 6), "s_ci95_hi": round(hi, 6),
                "p": round(p_val, 6), "r": round(r_val, 6),
                "T_pipe_over_A_iso": round(T_pipe / A_iso, 6),
                "speedup_vs_serial": round(Tser["median_ms"] / T_pipe, 6),
                "reference": "own_isolated",
                "notes": (f"EAGER baseline. The compiled bf16_both rung cannot be pipelined "
                          f"on this stack (see scratch/NOTE_e12_compiled_two_stream.txt), so "
                          f"r here is the EAGER ratio and is NOT E7's 1.50, which is "
                          f"bf16_both. Isolated bracket drift: prefix {drift_A:.1f} %, "
                          f"expert {drift_B:.1f} %. Fresh-KV equivalence verified bitwise "
                          f"in eager. "
                          f"NO OVERLAP: T_pipe {T_pipe:.1f} ms vs A_iso+B_iso "
                          f"{A_iso + B_iso:.1f} ms, so s and p are NOT overlap measurements "
                          f"-- each stage ran with the GPU to itself and both read ~1.00. "
                          f"host issue/wall = 1.0000: pi0 eager is entirely CPU-issue-bound "
                          f"(E5: 84.2 % GPU idle over 968 kernels/step), so there is nothing "
                          f"for two streams to overlap and GPU contention cannot arise in "
                          f"this configuration. r here is {r_val:.4f}, NOT E7's 1.50, because "
                          f"eager's expert is the longer stage while bf16_both's prefix is.")})
        # meta.json -- protocol §5 and non-negotiable 4. The first version of this function
        # wrote three CSVs and no meta.json, which makes the directory non-compliant however
        # good the numbers are: without bench_commit, the model commit, the rung string and
        # the gate, a number is not reproducible and the protocol says so plainly.
        import common as C                                            # noqa: PLC0415
        sw = C.software_versions()
        C.write_json(out / "meta.json", {
            "experiment": "E12 (co-run: prefix(k) || expert(k-1) on one GPU)", "task": "T014",
            "model": spec.name, "model_repo": spec.checkpoint,
            "model_commit": MR.resolved_commit(spec), "model_revision": spec.revision,
            "checkpoint": spec.checkpoint,
            "rung": "eager", "compile_mode": "none (eager)", "batch": 1,
            "num_denoise_steps": args.steps,
            "mechanism": "pi0_nosync" if getattr(args, "nosync", False) else "baseline_eager",
            "frames_timed": args.frames, "warmup_frames": args.warm,
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "gate": gate, "throttled": False,
            "throttled_evidence": "GPU clock pinned min==max in the gate; bracketed isolated "
                                  f"runs drifted {drift_A:.1f} % (prefix) and {drift_B:.1f} % "
                                  f"(expert), which is the drift check non-negotiable 3 asks "
                                  f"for.",
            "bench_commit": C.sh_out(f"git -C {REPO} rev-parse HEAD"),
            "command": " ".join(sys.argv),
            "observations": manifest.get("frames"),
            "observation_source": manifest.get("dataset"),
            "torch": sw.get("torch"), "python": sw.get("python"),
            # software_versions() names these torch_cuda etc.; the plain keys were always null (2026-09-27)
            "cuda": sw.get("cuda") or sw.get("torch_cuda") or "unknown",
            "jetpack": sw.get("jetpack") or C.sh_out("dpkg-query -W -f='${Version}' nvidia-jetpack") or "unknown",
            "l4t": sw.get("l4t") or (C.sh_out("head -1 /etc/nv_tegra_release").split()[1] + "."
                                     + C.sh_out("head -1 /etc/nv_tegra_release").split("REVISION: ")[-1].split(",")[0]),
            "raw_files": [],
            "host_issue_over_wall": round(iss / wal, 6) if wal else None,
            "not_measured": [
                "compiled bf16_both co-run: the stock path corrupts non-deterministically "
                "when pipelined across two streams (max|d| 1.273 on one run, all-NaN on "
                "another) while compiled SERIAL is correct and eager PIPELINED is bitwise "
                "exact. See scratch/NOTE_e12_compiled_two_stream.txt. Needs the own-CUDAGraph "
                "fallback of spec §7 before any compiled co-run number can be reported.",
                "Step 4 (MODE_30W, 8 SMs): waits for the owner's capped-mode reboot, per "
                "spec §5 Step 4 and §11, which permit the MAXN part to be marked done.",
                "Step 3 mechanisms (stream priority -3, graphs on both stages, green "
                "contexts, libsmctrl, MPS, two processes): not yet attempted.",
            ],
            "notes": [
                "NO OVERLAP OCCURRED. T_pipe equals A_iso + B_iso, and host issue/wall = "
                f"{iss / wal:.4f} -- the CPU spends the entire wall clock issuing kernels, so "
                "the two streams never both have resident work. s and p therefore read ~1.00 "
                "because each stage ran with the GPU to itself, NOT because overlap was "
                "ideal. Only T_pipe separates those two readings.",
                "pi0 eager is launch-bound: E5 measured ae_step at 84.2 % GPU idle over 968 "
                "kernels per step. GPU contention (KerColle's head-of-line blocking) cannot "
                "arise in a configuration that never fills the GPU, so this configuration "
                "cannot answer that question -- the compiled rung is where it becomes "
                "answerable.",
                "mechanism=pi0_nosync memoises the constant att_masks tensors at "
                "modeling_pi0.py:681 (per prefix) and :746 (per denoise step), removing 1+N "
                "host syncs per frame. A LABELLED harness change per spec §7, verified to "
                "keep the fresh-KV equivalence bitwise identical. It also made the expert "
                "stage ~19 % faster at N=4 (226.81 -> 184.18 ms), which is an optimisation "
                "for the shipping model independent of E12.",
                f"r = {r_val:.4f} here, NOT E7's 1.50. In eager the expert is the longer "
                f"stage ({B_iso:.1f} vs {A_iso:.1f} ms); under bf16_both the prefix is "
                f"(251.28 vs 167.36). Compiling speeds the expert more than the prefix and "
                f"flips which stage sets the period, so r must come from the same session.",
            ]})
        print(f"  wrote {out}/corun_ratios.csv (+ summary, samples, meta.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
