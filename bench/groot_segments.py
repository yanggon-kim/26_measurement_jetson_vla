"""T012 / E11a for GR00T: one DiT block, segment by segment.

GR00T lives in `~/venvs/groot` with its own codebase, so it cannot go through
`bench/host_segments.py`'s `main()` — that loads a LeRobot policy through
`model_registry`, and `MR.get("groot_n17")` raises by design rather than guessing wrap
points. But `host_segments`' **engine** is model-independent and imports cleanly in the
GR00T venv (checked, not assumed), so this driver reuses it rather than growing a second
timing harness whose numbers would not be comparable:

    time_segment    protocol §13.1's three timing modes
    make_style      eager / compiled / cudagraph, with the failure paths
    count_kernels   kernels per segment, from a torch profiler trace
    ModelInputs     the §13.8 emitter, whose header the in-place checker validates

What is GR00T-specific lives in two places and nowhere else: the loader (shared with
`bench/groot_phases.py`, so both measure the same model the same way) and
`bench/segments_groot.py` (the block's capture and its segment list).

## Which block, and why two of them

`DiT.forward` (:310-326) alternates: odd indices get `encoder_hidden_states=None` and are
pure self-attention, even indices cross-attend onto the VLM's tokens. π0's expert layers
are all the same kind, so π0's E11a measures layers 0 and 9 as a same-shape sanity check.
Here the two kinds are structurally different and both must be measured — reporting one
and calling it "the block" would hide half the architecture.

    ~/venvs/groot/bin/python bench/groot_segments.py \\
        --outdir results/<date>_E11a_groot_n17_hostseg --version n17
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segments_groot as SG                                          # noqa: E402
import host_segments as HS                                          # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions, write_json,  # noqa: E402
                    write_summary)
from host_segments import ModelInputs                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--version", default="n17")
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--gr00t-root", default=None)
    ap.add_argument("--api", choices=("new", "old"), default="new")
    ap.add_argument("--data-config", default="oxe_droid")
    ap.add_argument("--embodiment-tag", default="oxe_droid_relative_eef_relative_joint")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=1000)
    ap.add_argument("--styles", default="eager,compiled,cudagraph")
    ap.add_argument("--blocks", default="0,1,16,17",
                    help="DiT block indices. Default: block 0 and a middle block of EACH kind -- N1.7's "
                         "32 blocks alternate cross-attention (even) and self-attention (odd); the kind "
                         "each block ran as is read from the capture, not assumed")
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--mode", choices=("time", "count", "nsys"), default="time",
                    help="as bench/host_segments.py: `count` (root, CUPTI) writes kernels_per_call.csv and "
                         "the profiler traces; `nsys` captures 10 host sequences per style under "
                         "bench/run_nsys.sh; `time` does the timing, with --kernels-csv from `count`")
    ap.add_argument("--kernels-csv", default=None)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    styles = [s for s in args.styles.split(",") if s]
    kernels_map = HS.load_kernels_map(args.kernels_csv)

    # The loader is groot_phases', so this measures the same model the phase split did.
    import groot_phases as GP                                        # noqa: PLC0415
    g = GP.gate()
    print("gate:", json.dumps(g), flush=True)
    if g["gpu_cur_freq_hz"] != g["gpu_max_freq_hz"]:
        raise SystemExit(f"clocks not locked: {g}; run sudo jetson_clocks first")
    torch.set_float32_matmul_precision("high")
    # the observation VALUES come from np.random (groot_phases.build_observation); latency does not depend
    # on them, but seeding makes the captured tensors reproducible
    np.random.seed(1234)

    policy, ckpt_dir, model, inputs, n, shapes = GP.load_for_segments(args)
    blocks = model.action_head.model.transformer_blocks
    print(f"{len(blocks)} DiT blocks, N={n}", flush=True)

    want = [int(x) for x in args.blocks.split(",") if x.strip()]
    bad = [b for b in want if b >= len(blocks)]
    if bad:
        raise SystemExit(f"--blocks {bad} out of range: this model has {len(blocks)} "
                         f"DiT blocks (0..{len(blocks) - 1})")
    caps = {}
    for bi in want:
        with SG.DiTBlockCapture(model, bi) as cap, torch.no_grad():
            GP_step(model, inputs, args)
        if cap.args is None:
            raise SystemExit(
                f"DiT block {bi} was never called. DiT.forward iterates every block, so "
                f"this means the denoising pass did not run -- refusing to build segments "
                f"from nothing.")
        caps[bi] = cap
        print(f"  block {bi}: {cap.kind}", flush=True)
    kinds = {bi: c.kind for bi, c in caps.items()}
    attn_every = getattr(model.action_head.model, "attend_text_every_n_blocks", None)
    cross_onto = {bi: ("non-image (text) tokens" if bi % (2 * attn_every) == 0 else "image tokens")
                  for bi, k in kinds.items() if k == "cross_attn" and attn_every}
    with SG.StepExtrasCapture(model) as xcap, torch.no_grad():
        GP_step(model, inputs, args)
    torch.cuda.synchronize()

    cfg = model.action_head.model.config
    H = cfg.num_attention_heads
    HD = cfg.attention_head_dim
    D = H * HD
    SG._API[0] = args.api
    method_notes = []
    _ac_note = SG.model_autocast_ctx(args.api)[1]
    if _ac_note:
        method_notes.append(_ac_note)

    t0 = SG.intermediates(caps[want[0]])
    S = t0["x"].shape[1]
    P = next((c.args["encoder_hidden_states"].shape[1] for c in caps.values()
              if c.args["encoder_hidden_states"] is not None), "")
    prov = GP._provenance(args, g)
    model_name = f"groot_{args.version}"
    mi = ModelInputs(model_name, prov["model_commit"], sh_out(f"git -C {REPO} rev-parse HEAD"),
                     "MAXN+jetson_clocks", P, S, n)
    seg_rows, all_rows, layer_rows, samples, notes = [], [], [], [], []
    SEQ_EXCLUDE = ("attention_fused", "feedforward")     # attention + large linears are GPU work (§13.2)

    for bi, cap in caps.items():
        t = SG.intermediates(cap)
        ehs = t["encoder_hidden_states"]
        I = getattr(cfg, "ffn_dim", None) or 4 * D      # diffusers FeedForward default
        T_keys = ehs.shape[1] if ehs is not None else S
        extra = f";kind={cap.kind}"
        print(f"\nblock {bi} ({cap.kind}): x{tuple(t['x'].shape)} S={S} D={D} "
              f"I={I} H={H} head_dim={HD} T={T_keys}", flush=True)
        segs = SG.build_segments(t, S, D, I, H, HD, T_keys)
        for sd in segs:
            seg_rows.append({
                "model": model_name, "segment": sd["segment"], "order_in_layer": sd["order"],
                "layer_kind": cap.kind, "block": bi, "source_ref": sd["source_ref"],
                "ops": sd["ops"], "inputs": sd["inputs"], "outputs": sd["outputs"],
                "bytes_read": sd["bytes_read"], "bytes_written": sd["bytes_written"],
                "flops": sd["flops"], "flops_formula": sd["flops_formula"],
                "bytes_formula": sd.get("bytes_formula", ""), "in_model_code": sd["in_model"],
                "large_linear_on_S": "yes" if sd["segment"] == "feedforward" else "no",
                "in_host_sequence": "no" if sd["segment"] in SEQ_EXCLUDE else "yes",
                "notes": sd["notes"]})
            HS.measure_segment_styles(sd, bi, styles, args, out, kernels_map, mi, all_rows, samples,
                                      notes, variant_extra=extra)
            mi.add("bytes_host_segment", sd["bytes_read"] + sd["bytes_written"], "bytes", "value", 1,
                   "segments.csv", key=sd["segment"], variant=f"block={bi}{extra}", timing="analytic")
            mi.add("flops_host_segment", sd["flops"], "flops", "value", 1, "segments.csv",
                   key=sd["segment"], variant=f"block={bi}{extra}", timing="analytic")

        if not args.skip_sequence and args.mode in ("time", "nsys"):
            seq_segs = [s_ for s_ in segs if s_["segment"] not in SEQ_EXCLUDE]

            def seq(_segs=seq_segs):
                # results RETURNED: a compiled callable whose outputs are discarded is DCE'd
                return tuple(s_["fn"]() for s_ in _segs)

            if args.mode == "nsys":
                HS.nsys_capture_sequences(seq, bi, args)
                continue
            HS.measure_layer_sequence(seq, {s_["segment"] for s_ in seq_segs}, bi, styles, args, mi,
                                      layer_rows, all_rows, notes, variant_extra=extra,
                                      n_seg=len(seq_segs))

    if args.mode == "nsys":
        return 0
    xdefs = SG.build_step_extras(model, xcap, n)
    if SG.model_autocast_ctx(args.api)[1]:
        def _ac(fn):
            def w():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return fn()
            return w
        xdefs = [(a, _ac(f), r, o, ll) for a, f, r, o, ll in xdefs]
    defs4 = [(name, fn, ref, ops) for name, fn, ref, ops, _ll in xdefs]
    if args.mode == "count":
        HS.count_extras_and_write(defs4, styles, out, all_rows)
        return 0

    print("\n=== per-step extras ===", flush=True)
    kmap = {k: v for k, v in kernels_map.items() if len(k) == 2}   # extras: (segment, style), as finish()
    xrows = HS.measure_extra_defs(defs4, args, mi, all_rows, notes, styles, kmap)
    large = {name: ll for name, _f, _r, _o, ll in xdefs}
    for r in xrows:
        r.update({"block": "", "bytes_formula": "", "in_host_sequence": "no",
                  "large_linear_on_S": "yes" if large.get(r["segment"]) else "no",
                  "notes": ("per-INFERENCE extra (runs once, before the denoising loop)"
                            if r["segment"] == "state_encoder" else r["notes"])
                  + ("; a large linear layer on the S tokens, NOT a host segment (T012 step 1)"
                     if large.get(r["segment"]) else "")})
    seg_rows += xrows
    n_mi = HS.write_e11a_tables(out, seg_rows, all_rows, layer_rows, samples, mi)

    # docs/reference_scripts/check_model_inputs.py is the research side's, run IN PLACE; its MODELS
    # set has groot_n17 but not groot_n16 / groot_n15. Recorded rather than worked around.
    schema_note = "none"
    try:
        chk = (REPO / "docs" / "reference_scripts" / "check_model_inputs.py").read_text()
        allowed = set(re.findall(r'"([a-z0-9_]+)"', re.search(
            r"MODELS\s*=\s*\{([^}]*)\}", chk).group(1)))
        if model_name not in allowed:
            schema_note = (f"model_inputs.csv names model={model_name!r}, which is NOT in "
                           f"check_model_inputs.py's MODELS set {sorted(allowed)}; the checker will "
                           f"reject it (research side's schema).")
            notes.append(schema_note)
    except (OSError, AttributeError):
        pass

    sw = software_versions()
    cmd = " ".join(sys.argv)
    write_json(out / "meta.json", {
        "experiment": "E11a (GR00T DiT blocks, T010 step 1 file set)", "task": "T012",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name, "checkpoint": args.checkpoint, "model_repo": args.checkpoint,
        "model_commit": prov["model_commit"], "checkpoint_revision": prov["checkpoint_revision"],
        "gr00t_root": args.gr00t_root or "default", "gr00t_root_resolved": prov["gr00t_root_resolved"],
        "precision": "bfloat16", "compile_mode": "per exec style (eager / torch.compile default / CUDA graph)",
        "batch": 1, "num_inference_timesteps": n, "num_denoise_steps": n,
        "prefix_tokens": P, "suffix_tokens": S,
        "blocks_measured": {str(k): v for k, v in kinds.items()},
        "cross_attention_attends_onto": {str(k): v for k, v in cross_onto.items()} or "none",
        "n_blocks": len(blocks), "hidden_dim": D, "heads": H, "head_dim": HD,
        "host_sequence_segments": [s_ for s_ in ("adaln_cond", "adalayernorm1", "pos_embed", "residual",
                                                 "layernorm3", "residual_ff")
                                   if any(r["segment"] == s_ for r in seg_rows)],
        "n_warmup": args.warmup, "n_timed": args.timed, "exec_styles": styles,
        "compile_seconds_per_callable": HS.COMPILE_S,
        "compile_seconds_total": round(sum(HS.COMPILE_S), 2),
        "kernels_csv": args.kernels_csv or "none (kernel counts not supplied)",
        "power_mode": g["power_mode"], "sm_count": g["sm_count"], "gpu_hz": g["gpu_cur_freq_hz"],
        "gate": g, "gate_end": prov["gate_end"], "throttled": prov["throttled"],
        "throttled_evidence": prov["throttled_evidence"],
        "jetson_clocks": "applied before the run; GPU locked min==max (checked at start)",
        "observation_shapes": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in shapes.items()},
        "observation_seed": "np.random.seed(1234) before the observation is built",
        "cuda": sw.get("torch_cuda") or "unknown", "torch": sw.get("torch") or "unknown",
        "triton": sw.get("triton") or "unknown", "python": sw.get("python") or "unknown",
        "venv": sw.get("executable") or "unknown",
        "bench_commit": prov["bench_commit"], "bench_files_sha256": sw.get("bench_files_sha256") or "none",
        "command": cmd, "model_inputs_rows": n_mi, "raw_files": [],
        "not_measured": notes or "none", "schema_note": schema_note,
        "notes": method_notes + [
            "The timing engine and file set are bench/host_segments.py's: measure_segment_styles, "
            "measure_layer_sequence, nsys_capture_sequences, measure_extra_defs, count_extras_and_write and "
            "write_e11a_tables are the same functions the LeRobot E11a runs call, so the numbers are "
            "comparable with pi0's rather than produced by a second harness.",
            "Both kinds of DiT block are measured, first and middle of each: AlternateVLDiT.forward "
            "(dit.py:381-403) runs odd blocks as self-attention and even blocks as cross-attention onto the "
            "backbone features (text tokens when idx % (2*attend_text_every_n_blocks) == 0, image tokens "
            "otherwise). Variants carry block=<i>;kind=<kind>.",
            "The attention is NOT split into score / softmax / context (one fused SDPA kernel); the "
            "feed-forward is reported whole. Both are excluded from the host sequence, as the Gemma "
            "models' attention matmuls and large linears are (protocol §13.2).",
            "Per-step extras re-run the model's own modules on arguments captured from one real denoising "
            "step (segments_groot.StepExtrasCapture); state_encoder runs once per inference, the rest once "
            "per step. Those that are large linear layers on the S tokens are marked large_linear_on_S=yes "
            "in segments.csv and are not host segments.",
            "achieved_bw_GBps / achieved_tflops are published from the cudagraph style's gpu_event time "
            "only, as for the LeRobot models (eager gpu_event brackets a dispatch-bound call).",
        ],
    })
    rows_by = {}
    for r in layer_rows:
        rows_by.setdefault(r["variant"], []).append(f"{r['exec_style']} {r['key']} {r['median_ms']} ms")
    write_summary(out, f"E11a - {model_name} DiT host segments (T012 step 1)", [
        ("What", f"{len(caps)} DiT blocks ({', '.join(f'{b}={k}' for b, k in kinds.items())}) x "
                 f"{len(styles)} exec styles, every segment timed four ways (wall_isolated gap 0 / 2 ms, "
                 f"gpu_event, wall_launch; n = {args.timed}); host sequence + whole-layer graph per block; "
                 f"{len(defs4)} per-step extras. P = {P}, S = {S}, N = {n}."),
        ("Host layer", "\n".join(f"* `{v}`: " + "; ".join(x) for v, x in rows_by.items()) or "not measured"),
        ("Files", "segments.csv, segment_summary.csv, host_layer_summary.csv, kernels_per_call.csv (count "
                  "pass), nsys/ (host sequences), raw/segment_samples.csv.zst, raw/profiler/, model_inputs.csv"),
    ], notes)
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({n_mi} model_inputs rows, {len(samples)} samples)", flush=True)
    return 0


def GP_step(model, inputs, args):
    """One inference, matching bench/groot_phases.py's step()."""
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    if args.api == "old":
        with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                    dtype=torch.bfloat16):
            return model.get_action(inputs)
    with torch.no_grad():
        return model.get_action(inputs)


if __name__ == "__main__":
    raise SystemExit(main())
