"""Protocol §9 memory floor for GR00T, so E6 exists for it too (T009).

WHY THIS IS A SEPARATE FILE, AND WHY IT COUNTS BYTES BUT NOT FLOPs

bench/floors_models.py derives floors for pi0.5 and SmolVLA by walking a PaliGemma-shaped
tree and applying pi0's FLOP formula -- linear layers over S suffix tokens plus self-attention
of S queries against P + S cached keys. **GR00T's action head is not that.** It is a DiT whose
`transformer_blocks` ALTERNATE self-attention and cross-attention (segments_groot.py:18: "the
blocks alternate ... DiT.forward sends encoder_hidden_states only on ..."), so a step's
arithmetic is self-attention over S action tokens, cross-attention of S queries against the
backbone's P keys, and the block MLPs -- a different expression with a different token pairing
in each term. Applying pi0's formula here would produce a number that looks authoritative and
is wrong, which is worse than no number.

So this file computes what is architecture-INDEPENDENT and refuses to guess the rest:

  COUNTED   parameter bytes per region, summed from the loaded model. A byte count is a sum
            over tensors and needs no architectural assumption. This yields floor_mem_ms and,
            with a measured region time, achieved GB/s and % of peak bandwidth -- which is
            precisely the half of the roofline a bandwidth-based PIM argument rests on.
  NOT       FLOPs. Left as 0 and labelled NOT COUNTED, so bench/roofline.py leaves every
  COUNTED   compute column EMPTY rather than printing 0.0 -- a 0 there reads downstream as
            "this region does no arithmetic", which is the exact error that kept pi0's vision
            region out of the roofline until bench/vision_flops.py was written.

Deriving GR00T's DiT FLOPs properly is a real piece of work and is named as outstanding in
the report rather than approximated here.

Only N1.7 is loadable in this environment: N1.5 and N1.6 live on other gr00t branches with
their own venvs (model_registry.py records `installed: True` for n17 alone), so running them
means switching branches. Their floors are outstanding for the same reason.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

SPEC_GBPS, SPEC_TFLOPS = 204.8, 42.0
MEASURED_GBPS, MEASURED_TFLOPS = 187.47, 33.8506


def nbytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--version", default="n17")
    ap.add_argument("--api", default="new")
    ap.add_argument("--embodiment-tag", dest="embodiment_tag",
                    default="oxe_droid_relative_eef_relative_joint")
    ap.add_argument("--gr00t-root", dest="gr00t_root", default="")
    ap.add_argument("--compat-shims", action="store_true")
    ap.add_argument("--data-config", default="oxe_droid",
                    help="N1.5's old Gr00tPolicy API needs it (groot_phases._load_n15)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    import groot_phases as GP                                          # noqa: PLC0415
    policy, ckpt_dir, model, inputs, n, shapes = GP.load_for_segments(args)
    print(f"  loaded {args.checkpoint} (N={n})")

    head = model.action_head
    backbone = model.backbone
    regions, paths = {}, {}

    def take(key, mod, path):
        if mod is None:
            raise SystemExit(
                f"cannot resolve {key} at {path}; refusing to record 0 bytes for a region "
                f"that exists -- a zero is indistinguishable downstream from 'none'.")
        regions[key] = nbytes(mod)
        paths[key] = path

    take("action_head_dit", head, "model.action_head")
    take("backbone", backbone, "model.backbone")
    # Split the backbone into its vision and language parts where they are separable, so the
    # vision / vlm_prefix regions of GR00T's own phase split have byte counts to match.
    vis = lang = None
    for root, prefix in ((backbone, "model.backbone"),):
        for name, child in root.named_modules():
            low = name.lower()
            if vis is None and low.endswith(("vision_model", "vision_tower")):
                vis, vpath = child, f"{prefix}.{name}"
            if lang is None and low.endswith(("language_model", "text_model", "llm")):
                lang, lpath = child, f"{prefix}.{name}"
    if vis is not None:
        take("backbone_vision", vis, vpath)
    if lang is not None:
        take("backbone_language", lang, lpath)
    for k, v in regions.items():
        print(f"  {k:22s} {v/1e6:10.1f} MB   <- {paths[k]}")

    rows = []

    def add(region, b, note):
        for src, gbps, tfl in (("spec", SPEC_GBPS, SPEC_TFLOPS),
                               ("measured", MEASURED_GBPS, MEASURED_TFLOPS)):
            fmem = b / (gbps * 1e9) * 1e3
            rows.append({"model": f"groot_{args.version}", "region": region,
                         "bytes_read": b, "flops": 0, "peak_GBps": gbps,
                         "peak_TFLOPps": tfl, "floor_mem_ms": round(fmem, 4),
                         "floor_compute_ms": "", "floor_ms": round(fmem, 4),
                         "bound": "memory (compute floor NOT computed)",
                         "peak_source": src, "note": note})

    NOTFLOPS = ("FLOPs NOT COUNTED: GR00T's action head is a DiT whose transformer_blocks "
                "alternate self- and cross-attention, so pi0's linear+self-attention formula "
                "does not apply and applying it would give an authoritative-looking wrong "
                "number. bytes are a sum over tensors and need no architectural assumption.")
    if "backbone_vision" in regions:
        add("vision", regions["backbone_vision"],
            "backbone vision tower weights, read once per inference. " + NOTFLOPS)
    add("vlm_backbone", regions["backbone"],
        "whole backbone weights, matching GR00T's own `vlm_backbone` phase. " + NOTFLOPS)
    add("ae_step", regions["action_head_dit"],
        "action-head (DiT) weights read by ONE denoising step. The KV/context the DiT "
        "cross-attends to is NOT added here: its size depends on the backbone's prefix "
        "length, which this script does not measure. So this is a LOWER BOUND on the bytes "
        "an ae_step reads. " + NOTFLOPS)

    with (out / "floor.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "floor_inputs.json").write_text(json.dumps({
        "model": f"groot_{args.version}", "checkpoint": args.checkpoint,
        "checkpoint_dir": str(ckpt_dir), "embodiment_tag": args.embodiment_tag,
        "N": n, "region_parameter_bytes": regions, "resolved_paths": paths,
        "observation_shapes": {k: list(v) if hasattr(v, "__iter__") else v
                               for k, v in (shapes or {}).items()},
        "flops_counted": False, "flops_not_counted_reason": NOTFLOPS,
        "ae_step_bytes_is_lower_bound": True,
        "peaks": {"measured_GBps": MEASURED_GBPS, "spec_GBps": SPEC_GBPS},
    }, indent=2) + "\n")
    print(f"\n  {'region':14s} {'bytes':>14s} {'floor_mem_ms':>13s}")
    for r in rows:
        if r["peak_source"] != "measured":
            continue
        print(f"  {r['region']:14s} {r['bytes_read']:>14d} {r['floor_mem_ms']:>13.4f}")
    print(f"  wrote {out / 'floor.csv'}  (FLOPs deliberately not counted; see the note column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
