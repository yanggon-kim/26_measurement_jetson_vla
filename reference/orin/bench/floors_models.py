"""Protocol §9 floors for pi0.5 and SmolVLA, so E6 can be built for them (T008).

WHY THIS EXISTS
`floor.csv` -- the analytic bytes and FLOPs per region that E6's roofline stands on -- exists
only for pi0, written by bench/run_e3.py. Its floor_rows() hardcodes pi0's architecture: 18
layers, 1 KV head, head_dim 256, 8 attention heads, and the module path
`paligemma_with_expert.paligemma.model.vision_tower`. pi0.5 and SmolVLA differ in all of those,
so T008's `reports/E6_<m>_roofline.md` could not be produced. bench/regen_floor.py cannot help:
it re-derives floors from a run's recorded `region_parameter_bytes` and `tokens_detail`, and
those two fields are absent from every non-pi0 run's meta.json (checked, 2026-09-25).

WHAT IT REFUSES TO DO
Every architecture constant is read from the LOADED model's own config, and every region is
resolved to a real submodule by walking the container. If a region cannot be resolved this
script RAISES rather than recording 0 bytes. That matters because pi0's own floor.csv carries
`flops = 0` for `vision` with the note "FLOPs not counted analytically", and the consequence
was that vision silently had no operational intensity and dropped out of the roofline until
bench/vision_flops.py was written to fill it. A zero that means "not computed" is
indistinguishable downstream from a zero that means "genuinely none", and the roofline divides
by it.

TOKENS are measured, never assumed: pi0_common.token_counts() builds the real prefix through
the model's own embed_prefix, which is why it handles SmolVLA putting the robot state in the
PREFIX while pi0 puts it in the suffix.

GR00T is NOT handled here. It is a different family (a DiT action head, not a Gemma expert
behind PaliGemma), its regions are named differently in its own phase split
(`vlm_backbone`, `vlm_prefix_excl_vision`), and pretending one function covers both would
produce a floor that looks authoritative and is wrong. T009 needs its own pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

import pi0_common as P          # noqa: E402

SPEC_GBPS, SPEC_TFLOPS = 204.8, 42.0
MEASURED_GBPS, MEASURED_TFLOPS = 187.47, 33.8506


def _find(root, names: tuple[str, ...], what: str):
    """First submodule whose attribute path matches one of `names`. Raises if none."""
    for n in names:
        cur, ok = root, True
        for part in n.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur, n
    avail = [k for k, _ in root.named_children()]
    raise SystemExit(
        f"cannot resolve {what} on {type(root).__name__}: tried {names}, children are "
        f"{avail}. Refusing to record 0 bytes for a region that exists -- a zero here is "
        f"indistinguishable downstream from 'genuinely none' and the roofline divides by it.")


def nbytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters())


def regions_for(policy, spec) -> dict:
    """Region -> (module, resolved path). Reported so the caller can see what was counted."""
    container = getattr(policy.model, spec.container)
    vlm_root, _ = _find(container, ("paligemma.model", "vlm.model", "vlm", "paligemma"),
                        "the VLM root")
    vision, vp = _find(vlm_root, ("vision_tower", "vision_model", "vision_tower.vision_model"),
                       "the vision tower")
    lm, lp = _find(vlm_root, ("language_model", "text_model", "model.language_model"),
                   "the VLM language model")
    expert_layers, _ = _find(container, (spec.expert_layers_path,), "the expert layers")
    proj = None
    for cand in ("multi_modal_projector", "connector", "mm_projector"):
        if hasattr(vlm_root, cand):
            proj = getattr(vlm_root, cand)
            break
    return {"vision": (vision, vp), "projector": (proj, "multi_modal_projector"),
            "vlm_language_model": (lm, lp),
            "action_expert_layers": (expert_layers, spec.expert_layers_path)}


def arch_of(lm, expert_layers=None) -> dict:
    """Layers / heads / head_dim from the LOADED model, cross-checked against its config.

    The layer COUNT comes from the actual ModuleList, never from the config. SmolVLA prints
    "Reducing the number of VLM layers to 16" while its config still says
    num_hidden_layers = 32, so trusting the config doubled its KV-cache bytes and attention
    FLOPs -- a floor wrong by 2x that looks entirely plausible. The config describes the
    checkpoint; the module list describes what was loaded and what will actually run.
    """
    cfg = getattr(lm, "config", None)
    if cfg is None:
        raise SystemExit("the VLM language model has no .config; cannot read architecture")
    def g(*names, default=None):
        for n in names:
            if getattr(cfg, n, None) is not None:
                return getattr(cfg, n)
        return default
    layers = g("num_hidden_layers", "n_layer")
    heads = g("num_attention_heads", "n_head")
    kv = g("num_key_value_heads", default=heads)
    hd = g("head_dim")
    if hd is None:
        hidden = g("hidden_size", "n_embd")
        hd = hidden // heads if hidden and heads else None
    missing = [k for k, v in (("heads", heads), ("head_dim", hd)) if not v]
    if missing:
        raise SystemExit(f"config is missing {missing}; refusing to guess an architecture")

    actual = None
    for attr in ("layers", "model.layers", "h"):
        cur, ok = lm, True
        for part in attr.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            try:
                actual = len(cur)
            except TypeError:
                actual = None
            if actual:
                break
    if actual is None:
        raise SystemExit(
            "cannot count the VLM's actual decoder layers; refusing to fall back to the "
            "config, which SmolVLA is known to disagree with after truncation")
    note = ""
    if layers and int(layers) != actual:
        note = (f"config says num_hidden_layers={int(layers)} but the LOADED model has "
                f"{actual} -- using {actual}. SmolVLA truncates its VLM at load time.")
        print(f"  *** {note}")
    out = {"layers": actual, "config_layers": int(layers) if layers else None,
           "layers_source": "counted from the loaded module list", "layers_note": note,
           "heads": int(heads), "kv_heads": int(kv), "head_dim": int(hd)}
    if expert_layers is not None:
        try:
            out["expert_layers"] = len(expert_layers)
        except TypeError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("pi0", "pi05", "smolvla"))
    ap.add_argument("--rung", default="",
                    help="apply this opt_ladder rung BEFORE counting bytes. Rungs such as "
                         "bf16_both cast the vision tower and layernorms to bf16, so the bytes "
                         "a region moves -- and therefore its floor -- depend on the rung. A "
                         "compiled-rung roofline built on eager-load bytes would overstate "
                         "achieved bandwidth for every cast region.")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    import model_registry as MR                                        # noqa: PLC0415
    spec = MR.get(args.model)
    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    tokens = P.token_counts(policy, batch, spec=spec)
    Ptok, Stok = tokens["prefix_tokens_P"], tokens["suffix_tokens_S"]
    rung_note = "default load (no rung applied)"
    if args.rung:
        # Tokens are taken above, BEFORE the rung: the rung compiles embed_image and a token
        # count must not pay a compile. The rung's casts are what change the bytes counted
        # below, and torch.compile is lazy, so nothing is compiled by applying it here.
        import opt_ladder as OL                                        # noqa: PLC0415
        compiled, prep, prec, _, fails, _ = OL.apply_rung(policy, spec, args.rung)
        if fails:
            raise SystemExit(f"rung {args.rung} failed: {fails}")
        rung_note = f"{args.rung}: prep=[{prep}] compiled={compiled} matmul={prec}"
        print(f"  rung applied before counting bytes: {rung_note}")

    regs = regions_for(policy, spec)
    for k, (mod, path) in regs.items():
        print(f"  {k:22s} -> {path if mod is not None else 'ABSENT'}"
              + (f"  {nbytes(mod)/1e6:.1f} MB" if mod is not None else ""))
    lm = regs["vlm_language_model"][0]
    arch = arch_of(lm, expert_layers=regs["action_expert_layers"][0])
    print(f"  architecture (from the model's own config): {arch}")
    print(f"  tokens measured: P={Ptok}  S={Stok}")

    vis_b = nbytes(regs["vision"][0]) + (nbytes(regs["projector"][0])
                                         if regs["projector"][0] is not None else 0)
    lm_b = nbytes(lm)
    ae_b = nbytes(regs["action_expert_layers"][0])
    el_size = next(lm.parameters()).element_size()

    L, H, KV, HD = arch["layers"], arch["heads"], arch["kv_heads"], arch["head_dim"]
    # The expert has its OWN layer count. For pi0/pi0.5 it equals the VLM's, but that is a
    # fact about those models, not a rule, and the KV cache one expert step reads is indexed
    # by the layers that actually run.
    LE = arch.get("expert_layers", L)
    kv_bytes = 2 * LE * KV * HD * Ptok * el_size
    T = Ptok + Stok
    vlm_flops = 2 * Ptok * (lm_b // el_size)
    ae_lin = 2 * Stok * (ae_b // el_size)
    ae_attn = LE * 2 * (2 * H * Stok * T * HD)

    rows = []

    def add(region, b, f, note):
        for src, gbps, tfl in (("spec", SPEC_GBPS, SPEC_TFLOPS),
                               ("measured", MEASURED_GBPS, MEASURED_TFLOPS)):
            fmem = b / (gbps * 1e9) * 1e3
            fcomp = f / (tfl * 1e12) * 1e3
            rows.append({"model": args.model, "rung": args.rung or "default",
                         "region": region, "bytes_read": b, "flops": f,
                         "peak_GBps": gbps, "peak_TFLOPps": tfl,
                         "floor_mem_ms": round(fmem, 4),
                         "floor_compute_ms": round(fcomp, 4),
                         "floor_ms": round(max(fmem, fcomp), 4),
                         "bound": "memory" if fmem >= fcomp else "compute",
                         "peak_source": src, "note": note})

    add("vision", vis_b, 0,
        "vision tower + projector weights, read once per inference. FLOPs 0 = NOT COUNTED "
        "analytically here (pi0's equivalent is bench/vision_flops.py); a roofline must not "
        "read this as zero arithmetic.")
    add("vlm_prefix", lm_b, vlm_flops,
        f"VLM language-model weights once; FLOPs = 2 x P({Ptok}) x params "
        f"(element_size {el_size} B)")
    add("ae_step", ae_b + kv_bytes, ae_lin + ae_attn,
        f"expert layer weights {ae_b} B + KV cache {kv_bytes} B "
        f"(2 x {LE} expert layers x {KV} kv_heads x {HD} head_dim x P({Ptok}) x "
        f"{el_size} B); "
        f"FLOPs = linear 2 x S({Stok}) x params = {ae_lin/1e9:.2f} G plus attention "
        f"{LE} x 2 x (2 x {H} heads x S x T({T}) x {HD}) = {ae_attn/1e9:.2f} G")

    with (out / "floor.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "floor_inputs.json").write_text(json.dumps({
        "model": args.model, "rung": args.rung or "default", "rung_note": rung_note,
        "checkpoint": spec.checkpoint, "revision": spec.revision,
        "tokens": tokens, "architecture_from_config": arch,
        "element_size_bytes": el_size,
        "resolved_region_paths": {k: (v[1] if v[0] is not None else None)
                                  for k, v in regs.items()},
        "region_parameter_bytes": {"vision_plus_projector": vis_b,
                                   "vlm_language_model": lm_b,
                                   "action_expert_layers": ae_b,
                                   "kv_cache_per_step": kv_bytes},
        "peaks": {"measured_GBps": MEASURED_GBPS, "measured_TFLOPps": MEASURED_TFLOPS,
                  "spec_GBps": SPEC_GBPS, "spec_TFLOPps": SPEC_TFLOPS},
        "torch": torch.__version__,
    }, indent=2) + "\n")
    print(f"\n  {'region':12s} {'bytes':>14s} {'GFLOP':>10s} {'floor_ms':>9s}  bound")
    for r in rows:
        if r["peak_source"] != "measured":
            continue
        print(f"  {r['region']:12s} {r['bytes_read']:>14d} {r['flops']/1e9:>10.2f} "
              f"{r['floor_ms']:>9.4f}  {r['bound']}")
    print(f"  wrote {out / 'floor.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
