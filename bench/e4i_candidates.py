"""T005 / E4i: the best_bf16 search candidates beyond the ladder (the owner's request, 2026-10-22: "find the true best",
including attention backends on the language prefix and the action expert; numerical differences are acceptable when
the actions pass the equivalence check).

Imported by bench/rungs.py, which registers the candidates in rungs.EXTRA. Each candidate is a ladder rung plus
labelled preps:
  attn_<prefix>_<expert>  attention backend per part, impl in {eager, sdpa, fa2}. LeRobot forces "eager" before each
      call (modeling_pi0.py:845 / :917), so every attention module of the chosen part gets its own copy of the config
      with the chosen implementation. LeRobot's writes to the shared config no longer reach those modules.
      sdpa  transformers' sdpa_attention_forward, with LeRobot's additive fp32 mask cast to the query dtype (bf16). The
            fill value -2.38e38 is finite in bf16, so fully masked padding rows stay finite.
      fa2   flash_attn_func on the keys the mask keeps. Exact only for a padding-only mask: every unmasked query sees
            the same key set. That holds for the pi0 / pi0.5 prefix and the pi0.5 expert, NOT the pi0 expert, whose
            state token must not see the actions. It is checked on the first call, and the call raises if it does not
            hold. Padding query rows (unused) attend the valid keys instead of a uniform average.
  no_attn_fusion  Inductor's scaled-dot-product pattern rewrite (fx_passes/fuse_attention._sfdp_init) switched off,
      to measure what the rewrite contributes.
  autotune  forced GEMM autotuning (rungs.force_gemm_autotune).
"""
from __future__ import annotations

import copy

import torch

_REGISTERED = False


def _register():
    """Register thor_sdpa / thor_fa2 in transformers' AttentionInterface (once per process)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def thor_sdpa(module, query, key, value, attention_mask, **kw):
        if attention_mask is not None and attention_mask.dtype != query.dtype:
            attention_mask = attention_mask.to(query.dtype)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    def thor_fa2(module, query, key, value, attention_mask, scaling=None, **kw):
        from flash_attn import flash_attn_func
        if attention_mask is None:
            idx = None
        else:
            keep = attention_mask[0, 0] > -1e30                       # [Sq, Sk]: True = may attend
            valid_k = keep.any(dim=0)
            if not getattr(module, "_thor_fa2_checked", False):
                rows = keep[keep.any(dim=1)]                           # the queries that see anything
                if not bool((rows == valid_k[None, :]).all()):
                    raise RuntimeError("thor_fa2: mask is not padding-only (some query sees a different key set); "
                                       "flash-attn cannot represent it exactly")
                module._thor_fa2_checked = True
            idx = valid_k.nonzero().squeeze(1)
        k = key if idx is None else key.index_select(2, idx)
        v = value if idx is None else value.index_select(2, idx)
        out = flash_attn_func(query.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scaling,
                              causal=False)
        return out, None

    AttentionInterface.register("thor_sdpa", thor_sdpa)
    AttentionInterface.register("thor_fa2", thor_fa2)
    _REGISTERED = True


def _attn_modules(adapter, part):
    pg = getattr(adapter.policy.model, adapter.spec.container)
    if not hasattr(pg, "paligemma"):
        raise RuntimeError("attention backend switch implemented for pi0 / pi0.5 (PaliGemma + Gemma expert)")
    layers = pg.paligemma.model.language_model.layers if part == "prefix" else pg.gemma_expert.model.layers
    return [layer.self_attn for layer in layers]


def prep_attn(adapter, prefix="eager", expert="eager"):
    import rungs as RG
    if adapter.family != "lerobot" or adapter.name not in ("pi0", "pi05"):
        raise RG.NotApplicable("attention backend switch implemented for pi0 / pi0.5")
    if expert == "fa2" and adapter.name == "pi0":
        raise RG.NotApplicable("pi0's expert mask is not padding-only (the state token must not see the actions): "
                               "flash-attn cannot represent it exactly")
    _register()
    impl = {"eager": "eager", "sdpa": "thor_sdpa", "fa2": "thor_fa2"}
    notes = []
    for part, choice in (("prefix", prefix), ("expert", expert)):
        if choice == "eager":
            continue
        mods = _attn_modules(adapter, part)
        cfg = copy.copy(mods[0].config)
        cfg._attn_implementation = impl[choice]
        for m in mods:
            m.config = cfg
        notes.append(f"{part}: {len(mods)} attention modules -> {impl[choice]}")
    return "; ".join(notes) or "attention unchanged (eager)"


def prep_no_attn_fusion(adapter):
    import torch._inductor.fx_passes.fuse_attention as FA
    FA._sfdp_init = lambda: None
    return "Inductor's SDPA pattern rewrite (fuse_attention._sfdp_init) disabled"


def candidates():
    """Candidate rungs (registered in rungs.EXTRA by rungs.py). Built on the ladder's best LeRobot configuration
    (bf16 casts + graphs on the expert, vision and prefix) and its prefix-compile-without-graphs sibling."""
    import rungs as RG
    G, NG = RG.GRAPHS, RG.NOGRAPH
    base_g = [("ae", G, False), ("vis", G, True), ("vlm", G, True)]
    base_ng = [("ae", G, False), ("vis", G, True), ("vlm", NG, False)]
    out = []
    for pre, exp in (("sdpa", "eager"), ("eager", "sdpa"), ("sdpa", "sdpa"), ("fa2", "eager"), ("fa2", "sdpa"),
                     ("fa2", "fa2"), ("sdpa", "fa2")):
        for gname, tg in (("graphs", base_g), ("nvg", base_ng)):
            for at in (False, True):
                out.append(RG._r(f"bf16_mpad_{gname}_attn_{pre}_{exp}" + ("_autotune" if at else ""),
                                 f"bf16 casts + mpad + compile ({gname} on the prefix) + attention prefix={pre}, "
                                 f"expert={exp}" + (" + forced GEMM autotuning" if at else ""),
                                 "lerobot", "high", tg, ["eager_bf16", "linear_mpad", f"attn:{pre}:{exp}"], autotune=at,
                                 note="labelled variant: attention backend switched from LeRobot's forced eager",
                                 only=["pi0", "pi05"]))
    for gname, tg in (("graphs", base_g), ("nvg", base_ng)):
        out.append(RG._r(f"bf16_mpad_{gname}", f"bf16 casts + mpad + compile ({gname} on the prefix)",
                         "lerobot", "high", tg, ["eager_bf16", "linear_mpad"]))
        out.append(RG._r(f"bf16_mpad_{gname}_autotune", f"bf16 casts + mpad + compile ({gname}) + forced GEMM autotuning",
                         "lerobot", "high", tg, ["eager_bf16", "linear_mpad"], autotune=True))
        out.append(RG._r(f"bf16_mpad_{gname}_nofuse", f"bf16 casts + mpad + compile ({gname}), Inductor attention rewrite off",
                         "lerobot", "high", tg, ["eager_bf16", "linear_mpad", "no_attn_fusion"]))
    # stage 2 (after stage 1 showed forced autotuning speeds pi0's expert 79.8 -> 63.1 ms but slows vision and prefix):
    # autotune only the listed parts (mode max-autotune on them, the SM gate lifted, global max_autotune off)
    AT = "max-autotune"
    for name, tg in (("ae_at", [("ae", AT, False), ("vis", G, True), ("vlm", G, True)]),
                     ("ae_vlm_at", [("ae", AT, False), ("vis", G, True), ("vlm", AT, True)]),
                     ("ae_vis_at", [("ae", AT, False), ("vis", AT, True), ("vlm", G, True)])):
        out.append(RG._r(f"bf16_mpad_graphs_{name}", f"bf16 casts + mpad + graphs everywhere; GEMM autotuning on {name[:-3]} only",
                         "lerobot", "high", tg, ["eager_bf16", "linear_mpad"], autotune="parts"))
    out.append(RG._r("bf16_mpad_whole_graphs", "bf16 casts + mpad + sample_actions compiled with graphs (e2e only)",
                     "lerobot", "high", [("whole", G, False)], ["eager_bf16", "linear_mpad"], regions=False))
    out.append(RG._r("bf16_mpad_whole_graphs_autotune", "bf16_mpad_whole_graphs + forced GEMM autotuning (e2e only)",
                     "lerobot", "high", [("whole", G, False)], ["eager_bf16", "linear_mpad"], regions=False,
                     autotune=True))
    return out


def groot_candidates():
    """GR00T (already all-bf16, backbone attention already flash-attn 2): compile / graphs per part, forced GEMM
    autotuning, linear_mpad, and for N1.7 a dynamo setting that may let its vision tower compile
    (capture_scalar_outputs: the .item() feeding flash_attn_varlen's max_seqlen becomes a symbolic int instead of a
    graph break)."""
    import rungs as RG
    G, NG = RG.GRAPHS, RG.NOGRAPH
    ae, vis_ng, llm_ng, llm_g = ("ae", G, True), ("vis", NG, False), ("llm", NG, False), ("llm", G, True)
    ae_at, vis_at = ("ae", "max-autotune", True), ("vis", "max-autotune-no-cudagraphs", False)
    # stage 2 parts that must NOT autotune compile with mode "default": with the SM gate lifted, the ladder's no-graphs
    # mode (max-autotune-no-cudagraphs) would autotune their GEMMs too
    vis_d, llm_d = ("vis", "default", False), ("llm", "default", False)
    spec = [
        ("ae_graphs_mpad", [ae], ["linear_mpad"], False, None),
        ("ae_graphs_autotune", [ae], [], True, None),
        ("ae_graphs_mpad_autotune", [ae], ["linear_mpad"], True, None),
        ("llm_compile_autotune", [ae, llm_ng], [], True, None),
        ("llm_graphs_autotune", [ae, llm_g], [], True, None),
        ("llm_graphs_mpad_autotune", [ae, llm_g], ["linear_mpad"], True, None),
        ("vlm_graphs_autotune", [ae, vis_ng, llm_g], [], True, None),
        ("vlm_graphs_mpad_autotune", [ae, vis_ng, llm_g], ["linear_mpad"], True, None),
        ("all_compile_autotune_mpad", [ae, vis_ng, llm_ng], ["linear_mpad"], True, None),
        ("vlm_compile_scalar", [ae, vis_ng, llm_ng], ["capture_scalar"], False, ["groot_n17"]),
        # stage 2: autotune per part (mode max-autotune on the listed parts only)
        ("ae_at", [ae_at], [], "parts", None),
        ("ae_at_mpad", [ae_at], ["linear_mpad"], "parts", None),
        ("llm_graphs_ae_at", [ae_at, llm_g], [], "parts", None),
        ("llm_graphs_ae_at_mpad", [ae_at, llm_g], ["linear_mpad"], "parts", None),
        ("all_compile_ae_at", [ae_at, vis_d, llm_d], [], "parts", None),
        ("all_compile_ae_at_mpad", [ae_at, vis_d, llm_d], ["linear_mpad"], "parts", None),
        ("vlm_graphs_ae_at_mpad", [ae_at, vis_d, llm_g], ["linear_mpad"], "parts", None),
        ("all_compile_ae_vis_at_mpad", [ae_at, vis_at, llm_d], ["linear_mpad"], "parts", None),
        ("all_compile_autotune_scalar", [ae, vis_ng, llm_ng], ["capture_scalar"], True, ["groot_n17"]),
        ("vlm_graphs_autotune_scalar", [ae, vis_ng, llm_g], ["capture_scalar"], True, ["groot_n17"]),
    ]
    return [RG._r(k, f"GR00T E4i candidate: {', '.join(f'{p}@{m}' for p, m, _ in t)}"
                     + (f" + {'+'.join(pr)}" if pr else "")
                  + (" + GEMM autotuning on the max-autotune parts only" if at == "parts" else " + forced GEMM autotuning" if at else ""),
                  "groot", "high", t, pr, autotune=at, only=only) for k, t, pr, at, only in spec]
