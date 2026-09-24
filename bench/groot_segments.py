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
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segments_groot as SG                                          # noqa: E402
from common import REPO, collect_raw, sh_out, software_versions, write_json  # noqa: E402
from host_segments import (MI_HEADER, ModelInputs, count_kernels,    # noqa: E402
                           make_style, time_segment)


def summarize(xs):
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[0] if n == 1 else s[int(p * (n - 1))] + (
        s[min(int(p * (n - 1)) + 1, n - 1)] - s[int(p * (n - 1))]) * (
        p * (n - 1) - int(p * (n - 1)))
    return {"median_ms": statistics.median(s), "p5_ms": q(0.05), "p95_ms": q(0.95),
            "n": n}


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
    ap.add_argument("--blocks", default="",
                    help="comma-separated DiT block indices. Default: one self-attention "
                         "and one cross-attention block, chosen from what the model "
                         "actually ran — the two kinds alternate and are different")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    # The loader is groot_phases', so this measures the same model the phase split did.
    import groot_phases as GP                                        # noqa: PLC0415
    g = GP.gate()
    print("gate:", json.dumps(g), flush=True)
    if g["gpu_cur_freq_hz"] != g["gpu_max_freq_hz"]:
        raise SystemExit(f"clocks not locked: {g}; run sudo jetson_clocks first")
    torch.set_float32_matmul_precision("high")

    policy, ckpt_dir, model, inputs, n, shapes = GP.load_for_segments(args)
    blocks = model.action_head.model.transformer_blocks
    print(f"{len(blocks)} DiT blocks, N={n}", flush=True)

    # Which blocks: one of each kind. `interleave_self_attention` decides, but reading the
    # config would be an assumption — the capture records what it SAW, and the pair is
    # chosen from that.
    want = ([int(x) for x in args.blocks.split(",") if x.strip()]
            or [0, 1] if len(blocks) > 1 else [0])
    caps = {}
    for bi in want:
        with SG.DiTBlockCapture(model, bi) as cap, torch.no_grad():
            GP_step(model, inputs, args)
        if cap.args is None:
            raise SystemExit(
                f"DiT block {bi} was never called. DiT.forward iterates every block, so "
                f"this means the denoising pass did not run — refusing to build segments "
                f"from nothing.")
        caps[bi] = cap
        print(f"  block {bi}: {cap.kind}", flush=True)
    kinds = {bi: c.kind for bi, c in caps.items()}
    if len(set(kinds.values())) < 2 and len(blocks) > 1:
        print(f"  NOTE: both captured blocks are {next(iter(set(kinds.values())))}; "
              f"the self/cross alternation did not apply to {want}", flush=True)

    cfg = model.action_head.model.config
    H = cfg.num_attention_heads
    HD = cfg.attention_head_dim
    D = H * HD
    mi = ModelInputs(f"groot_{args.version}", args.checkpoint,
                     sh_out(f"git -C {REPO} rev-parse HEAD"),
                     g["power_mode"], "", "", n)
    styles = [s for s in args.styles.split(",") if s]
    seg_rows, all_rows, notes = [], [], []

    for bi, cap in caps.items():
        t = SG.intermediates(cap)
        S = t["x"].shape[1]
        ehs = t["encoder_hidden_states"]
        I = getattr(cfg, "ffn_dim", None) or 4 * D      # diffusers FeedForward default
        T_keys = ehs.shape[1] if ehs is not None else S
        print(f"\nblock {bi} ({cap.kind}): x{tuple(t['x'].shape)} S={S} D={D} "
              f"I={I} H={H} head_dim={HD} T={T_keys}", flush=True)
        for sd in SG.build_segments(t, S, D, I, H, HD, T_keys):
            row = {"model": f"groot_{args.version}", "segment": sd["segment"],
                   "order_in_layer": sd["order"], "layer_kind": cap.kind,
                   "block": bi, "in_model": sd["in_model"],
                   "source_ref": sd["source_ref"], "ops": sd["ops"],
                   "inputs": sd["inputs"], "outputs": sd["outputs"],
                   "bytes_read": sd["bytes_read"], "bytes_written": sd["bytes_written"],
                   "bytes_formula": sd["bytes_formula"], "flops": sd["flops"],
                   "flops_formula": sd["flops_formula"], "notes": sd["notes"]}
            for style in styles:
                fn, err = make_style(sd["fn"], style, f"{sd['segment']}@block{bi}")
                if fn is None:
                    notes.append(f"{sd['segment']} block{bi} {style}: {err}")
                    row[f"{style}_median_ms"] = ""
                    continue
                s = summarize(time_segment(fn, "wall_isolated", 20, args.timed))
                row[f"{style}_median_ms"] = round(s["median_ms"], 6)
                all_rows.append({"segment": sd["segment"], "block": bi,
                                 "layer_kind": cap.kind, "exec_style": style,
                                 "timing": "wall_isolated", **{
                                     k: round(v, 6) for k, v in s.items()
                                     if k != "n"}, "n": s["n"]})
                mi.add_stats(f"t_seg_{sd['segment']}", s, "ms", s["n"],
                             "host_segments.csv",
                             variant=f"block={bi};kind={cap.kind}", exec_style=style,
                             timing="wall_isolated")
                if style == "eager":
                    nk = count_kernels(fn)
                    row["kernels_eager"] = nk
                    mi.add(f"kernels_seg_{sd['segment']}", nk, "count", "median", 1,
                           "host_segments.csv",
                           variant=f"block={bi};kind={cap.kind}", exec_style=style)
            mi.add(f"bytes_seg_{sd['segment']}", sd["bytes_read"] + sd["bytes_written"],
                   "bytes", "analytic", 1, "host_segments.csv",
                   variant=f"block={bi};kind={cap.kind}")
            mi.add(f"flops_seg_{sd['segment']}", sd["flops"], "flops", "analytic", 1,
                   "host_segments.csv", variant=f"block={bi};kind={cap.kind}")
            seg_rows.append(row)
            print(f"  {sd['segment']:20s} "
                  + "  ".join(f"{s}={row.get(f'{s}_median_ms', '')}" for s in styles),
                  flush=True)

    with (out / "host_segments.csv").open("w", newline="") as f:
        keys = sorted({k for r in seg_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in seg_rows])
    with (out / "raw" / "segment_timings.csv").open("w", newline="") as f:
        keys = sorted({k for r in all_rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in all_rows])
    mi.write(out / "model_inputs.csv")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11a (GR00T DiT block)", "task": "T012",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": f"groot_{args.version}", "checkpoint": args.checkpoint,
        "model_repo": args.checkpoint, "precision": "bfloat16",
        "num_inference_timesteps": n, "blocks_measured": kinds,
        "n_blocks": len(blocks), "hidden_dim": D, "heads": H, "head_dim": HD,
        "n_warmup": args.warmup, "n_timed": args.timed, "exec_styles": styles,
        "power_mode": g["power_mode"], "gate": g,
        "gr00t_root": args.gr00t_root,
        "observation_shapes": {k: {kk: list(vv) for kk, vv in v.items()}
                               for k, v in shapes.items()},
        "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "command": " ".join(sys.argv),
        "raw_files": [],
        "not_measured": notes,
        "notes": [
            "Both kinds of DiT block are measured. DiT.forward (dit.py:310-326) sends "
            "encoder_hidden_states only on even indices when interleave_self_attention "
            "is set, so odd blocks are pure self-attention and even blocks cross-attend "
            "onto the VLM's tokens. Measuring one and calling it 'the block' would hide "
            "half the architecture.",
            "The attention is NOT split into score / softmax / context. attn1 is a "
            "diffusers Attention running under _sdpa_context(), so those happen inside "
            "one fused kernel; separating them would mean measuring a different "
            "implementation than the model's. pi0's eager Gemma path does expose them, "
            "which is why its segment table has a softmax row and this one does not.",
            "The feed-forward is reported whole. pi0's table splits act_fn(gate)*up from "
            "its projections per protocol §13.2; the diffusers FeedForward is a "
            "Sequential whose pieces would have to be indexed by position, which is "
            "brittle against a version change.",
            "The timing engine is bench/host_segments.py's, unchanged, so these numbers "
            "are comparable with pi0's rather than produced by a second harness.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}", flush=True)
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
