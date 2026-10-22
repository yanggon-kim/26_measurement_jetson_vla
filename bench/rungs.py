"""T005: the optimisation rungs (E4 Part B ladder, E4i search candidates) as ONE definition, shared by every caller.

Ported from reference/orin/bench/opt_ladder.py (LeRobot RUNGS, preps) and the flag combinations of
reference/orin/bench/groot_phases.py + bench/queues/queue_n1d6.sh / queue_groot_combined.sh (GR00T). T006-T014 put a
model into a named configuration (`best_bf16` included, via results/<date>_E4i_summary/best_bf16.json) through
apply_rung() and nothing else, so the configuration profiled is the configuration timed.

Changes from the Orin, each deliberate:
- GR00T's ladder is CUMULATIVE (task T005 step 2): vis_compile keeps the compiled head; vlm_compile keeps the compiled
  vision too (the Orin's separate `all_compile`). llm_compile / llm_graphs are the Orin's non-cumulative rungs (head +
  LLM, vision eager), kept for attribution and for a model whose vision cannot be compiled.
- GR00T's vision and LLM callables are the ones bench/regions.py times (Eagle: extract_feature and the decoder stack;
  Qwen3-VL: get_image_features and the decoder stack), so every region boundary stays outside a compiled graph.
- eager_fp32 is E3's (matmul "highest", TF32 off for cuBLAS AND cuDNN); every other rung uses matmul "high" with
  cuDNN TF32 at the torch default (on). The Orin's eager_fp32 left cuDNN TF32 at the default.
- eager_bf16 (new): the bf16 casts of bf16_both (LeRobot) with nothing compiled: the "eager bf16 path" the E4i
  equivalence check compares against. For a model already all-bf16 it is eager_tf32 under another name, and says so.
- The forced-autotune override (`autotune: True`) keeps ATEN in the backend list when Inductor accepts it, so the
  autotuner can still choose cuBLAS; the Orin restricted it to TRITON after an assertion (recorded per row).

A rung is a dict:
  key, label, family ("lerobot" | "groot"), precision ("highest" | "high"), preps [names], targets [(part, mode, clone)],
  regions (False: the rung compiles sample_actions as one unit and only e2e is measurable), autotune (bool),
  only (model names the rung applies to, or None), note.
Parts: LeRobot ae = model.denoise_step, vis = <container>.embed_image, vlm = <container>.forward,
       whole = model.sample_actions; GR00T ae = action_head.get_action, vis = the vision callable, llm = the decoder.
"""
from __future__ import annotations

import inspect
import time
import types

import torch

NOGRAPH = "max-autotune-no-cudagraphs"      # compile without CUDA graphs (the stock mode that separates the two)
GRAPHS = "reduce-overhead"                  # compile + CUDA graphs


class NotApplicable(Exception):
    """The rung cannot exist for this model (as distinct from failing): the row says why."""


# ------------------------------------------------------------------------------------------------ LeRobot preps
def _dominant_dtype(policy):
    """The dtype most parameter ELEMENTS are in (counting tensors misreads pi0: many small fp32 norms)."""
    from collections import Counter
    c = Counter()
    for p in policy.model.parameters():
        c[p.dtype] += p.numel()
    return c.most_common(1)[0][0] if c else None


def prep_bf16_base(policy):
    """B0: an fp32 checkpoint cast to bf16 through LeRobot's own to_bfloat16_for_selected_params (keeps its fp32
    keep-list: vision tower, projector, layer norms). pi0.5 ships fp32; pi0 and SmolVLA load bf16 already."""
    if _dominant_dtype(policy) == torch.bfloat16:
        raise NotApplicable("checkpoint already loads bf16 (dominant parameter dtype bfloat16), so there is no fp32 "
                            "base to cast")
    pg = getattr(policy.model, "paligemma_with_expert", None)
    fn = getattr(pg, "to_bfloat16_for_selected_params", None) if pg is not None else None
    if fn is None:
        raise NotApplicable(f"{type(policy).__name__} exposes no to_bfloat16_for_selected_params")
    fn("bfloat16")
    return "paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32)"


def prep_bf16_vision(policy):
    """B2: vision tower + projector to bf16 and embed_image's fp32 up-cast removed (modeling_pi0.py:441-450,
    modeling_pi05.py:438-447: identical bodies). SmolVLA has no fp32 vision island (not applicable)."""
    if type(policy).__name__ not in ("PI0Policy", "PI05Policy"):
        raise NotApplicable(f"{type(policy).__name__} has no fp32 vision island: LeRobot loads its vision tower in "
                            f"bf16 already")
    note = ""
    if _dominant_dtype(policy) != torch.bfloat16:
        note = prep_bf16_base(policy) + "; "
    pg = policy.model.paligemma_with_expert
    src = inspect.getsource(type(pg).embed_image)
    for token in ("get_image_features", "pooler_output", "torch.float32"):
        if token not in src:
            raise SystemExit(f"embed_image no longer contains {token!r}: re-derive prep_bf16_vision from source")
    pg.paligemma.model.vision_tower.to(torch.bfloat16)
    mmp = getattr(pg.paligemma.model, "multi_modal_projector", None)
    if mmp is not None:
        mmp.to(torch.bfloat16)

    def embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.bfloat16:
            image = image.to(torch.bfloat16)
        feats = self.paligemma.model.get_image_features(image)
        feats = feats.pooler_output * self.paligemma.config.text_config.hidden_size ** 0.5
        return feats.to(out_dtype) if feats.dtype != out_dtype else feats

    pg.embed_image = types.MethodType(embed_image, pg)
    return note + "vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed"


def prep_bf16_norms(policy):
    """B3: the layer norms of LeRobot's keep-list to bf16, own parameters only (recurse=False): pi0.5's adaptive
    RMSNorm embeds a conditioning Linear ('dense') fed from fp32 tensors created outside the model (Orin finding)."""
    if type(policy).__name__ not in ("PI0Policy", "PI05Policy"):
        raise NotApplicable(f"{type(policy).__name__} has no params_to_keep_float32: its norms are not held in fp32")
    note = ""
    if _dominant_dtype(policy) != torch.bfloat16:
        note = prep_bf16_base(policy) + "; "
    n = skipped = 0
    for name, mod in policy.model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("input_layernorm", "post_attention_layernorm") or name.endswith("model.norm"):
            own = list(mod.parameters(recurse=False))
            if not own:
                continue
            for p in own:
                p.data = p.data.to(torch.bfloat16)
            n += 1
            skipped += sum(1 for _ in mod.children())
    note += f"{n} layer-norm modules -> bf16 (own parameters only)"
    if skipped:
        note += f"; {skipped} submodule(s) inside them left as they were (pi0.5's adaptive-norm conditioning Linear)"
    return note


def prep_bf16_both(policy):
    return prep_bf16_vision(policy) + "; " + prep_bf16_norms(policy)


def prep_eager_bf16(policy):
    """The eager bf16 reference path: bf16_both's casts where they apply, nothing else."""
    try:
        return prep_bf16_both(policy)
    except NotApplicable as e:
        return f"no cast needed ({e}); this rung equals eager_tf32 for this model"


def prep_smolvlm_static_posids(policy):
    """SmolVLA only: replace SmolVLMVisionEmbeddings' boolean-mask index_put (a data-dependent op that cannot be
    stream-captured) by the identical static position ids. The mask is all-true by construction: embed_image passes
    patch_attention_mask=None (smolvlm_with_expert.py:179-186) and the transformer then builds torch.ones
    (transformers 5.3.0 modeling_smolvlm.py:355-363). Both facts are asserted."""
    import transformers.models.smolvlm.modeling_smolvlm as M
    if type(policy).__name__ != "SmolVLAPolicy":
        raise NotApplicable(f"SmolVLA-only prep; got {type(policy).__name__}")
    from lerobot.policies.smolvla import smolvlm_with_expert as SWE
    if "patch_attention_mask = None" not in inspect.getsource(SWE.SmolVLMWithExpertModel.embed_image):
        raise SystemExit("embed_image no longer passes patch_attention_mask=None: re-check the all-true-mask premise")
    if "torch.ones" not in inspect.getsource(M.SmolVLMVisionTransformer.forward):
        raise SystemExit("SmolVLMVisionTransformer no longer builds an all-ones mask")

    def forward(self, pixel_values, patch_attention_mask):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side,
                                  device=pixel_values.device)
        mh, mw = patch_attention_mask.size(1), patch_attention_mask.size(2)
        h = torch.arange(mh, device=pixel_values.device, dtype=torch.float32) * (1.0 / mh)
        w = torch.arange(mw, device=pixel_values.device, dtype=torch.float32) * (1.0 / mw)
        h = torch.clamp(h, max=(1.0 - 1e-6)).to(pixel_values.dtype)
        w = torch.clamp(w, max=(1.0 - 1e-6)).to(pixel_values.dtype)
        bh = torch.bucketize(h, boundaries, right=True)
        bw = torch.bucketize(w, boundaries, right=True)
        pos = (bh[:, None] * self.num_patches_per_side + bw[None, :]).reshape(1, -1).expand(batch_size, -1)
        return embeddings + self.position_embedding(pos)

    M.SmolVLMVisionEmbeddings.forward = forward
    return "SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids"


PREPS = {"bf16_base": prep_bf16_base, "bf16_vision": prep_bf16_vision, "bf16_norms": prep_bf16_norms,
         "bf16_both": prep_bf16_both, "eager_bf16": prep_eager_bf16, "static_posids": prep_smolvlm_static_posids}


# ------------------------------------------------------------------------------------------------ the rungs
def _r(key, label, family, precision, targets=(), preps=(), regions=True, autotune=False, only=None, note=""):
    return {"key": key, "label": label, "family": family, "precision": precision, "targets": list(targets),
            "preps": list(preps), "regions": regions, "autotune": autotune, "only": only, "note": note}


G, NG = GRAPHS, NOGRAPH
LEROBOT_RUNGS = [
    _r("eager_fp32", "eager, matmul highest, TF32 off (E3's baseline)", "lerobot", "highest"),
    _r("eager_tf32", "+ TF32 (matmul high, cuDNN TF32 on)", "lerobot", "high"),
    _r("eager_bf16", "eager with bf16_both's casts, nothing compiled (equivalence reference)", "lerobot", "high",
       preps=["eager_bf16"]),
    _r("bf16_base", "B0: fp32 checkpoint -> bf16 (LeRobot's own cast) on ae_graphs", "lerobot", "high",
       [("ae", G, False)], ["bf16_base"]),
    _r("ae_compile", "+ compile denoise_step (no CUDA graphs)", "lerobot", "high", [("ae", NG, False)]),
    _r("ae_graphs", "+ CUDA graphs on denoise_step", "lerobot", "high", [("ae", G, False)]),
    _r("vis_compile", "+ compile embed_image (no CUDA graphs)", "lerobot", "high", [("ae", G, False), ("vis", NG, False)]),
    _r("vis_static_pos", "+ static vision position ids (SmolVLA only; no new graph)", "lerobot", "high",
       [("ae", G, False), ("vis", NG, False)], ["static_posids"], only=["smolvla"]),
    _r("vis_graphs", "+ CUDA graphs on embed_image (output cloned)", "lerobot", "high",
       [("ae", G, False), ("vis", G, True)]),
    _r("vlm_compile", "+ compile container.forward (no CUDA graphs)", "lerobot", "high",
       [("ae", G, False), ("vis", G, True), ("vlm", NG, False)]),
    _r("vlm_compile_nvg", "compile container.forward, embed_image eager (alternative to vlm_compile)", "lerobot", "high",
       [("ae", G, False), ("vlm", NG, False)]),
    _r("vlm_graphs", "+ CUDA graphs on container.forward (output cloned)", "lerobot", "high",
       [("ae", G, False), ("vis", G, True), ("vlm", G, True)]),
    _r("whole_graphs", "compile sample_actions with CUDA graphs (LeRobot's own path); e2e only", "lerobot", "high",
       [("whole", G, False)], regions=False),
    _r("bf16_vision", "B2: + bf16 vision tower (on ae_graphs)", "lerobot", "high", [("ae", G, False)], ["bf16_vision"]),
    _r("bf16_norms", "B3: + bf16 layer norms (on ae_graphs)", "lerobot", "high", [("ae", G, False)], ["bf16_norms"]),
    _r("bf16_both", "B2 + B3 with ae graphs, vision graphs, prefix compiled", "lerobot", "high",
       [("ae", G, False), ("vis", G, True), ("vlm", NG, False)], ["bf16_both"]),
    _r("bf16_both_graphs", "B2 + B3 with CUDA graphs on the prefix too", "lerobot", "high",
       [("ae", G, False), ("vis", G, True), ("vlm", G, True)], ["bf16_both"]),
]
GROOT_RUNGS = [
    _r("eager_fp32", "eager, matmul highest, TF32 off (E3's baseline)", "groot", "highest"),
    _r("eager_tf32", "+ TF32 (matmul high, cuDNN TF32 on)", "groot", "high",
       note="GR00T loads all-bf16 (Gr00tPolicy), so this is also its eager bf16 path"),
    _r("ae_compile", "+ compile action_head.get_action (no CUDA graphs)", "groot", "high", [("ae", NG, True)]),
    _r("ae_graphs", "+ CUDA graphs on action_head.get_action (output cloned)", "groot", "high", [("ae", G, True)]),
    _r("vis_compile", "+ compile the vision callable (no CUDA graphs)", "groot", "high", [("ae", G, True), ("vis", NG, False)]),
    _r("vis_graphs", "+ CUDA graphs on the vision callable (output cloned)", "groot", "high", [("ae", G, True), ("vis", G, True)]),
    _r("vlm_compile", "+ compile the LLM decoder stack (vision compiled too: cumulative)", "groot", "high",
       [("ae", G, True), ("vis", NG, False), ("llm", NG, False)]),
    _r("vlm_graphs", "+ CUDA graphs on the decoder stack (output cloned)", "groot", "high",
       [("ae", G, True), ("vis", NG, False), ("llm", G, True)]),
    _r("all_compile_autotune", "vlm_compile + forced GEMM autotuning (Inductor's SM gate overridden; labelled)", "groot",
       "high", [("ae", G, True), ("vis", NG, False), ("llm", NG, False)], autotune=True),
    _r("llm_compile", "head graphs + decoder compiled, vision eager (the Orin's non-cumulative vlm_compile)", "groot",
       "high", [("ae", G, True), ("llm", NG, False)]),
    _r("llm_graphs", "head graphs + decoder graphs, vision eager", "groot", "high", [("ae", G, True), ("llm", G, True)]),
]
RUNGS = {"lerobot": {r["key"]: r for r in LEROBOT_RUNGS}, "groot": {r["key"]: r for r in GROOT_RUNGS}}
EXTRA = {"lerobot": {}, "groot": {}}           # E4i search candidates, registered by bench/e4i_candidates.py


def family_of(model):
    return "groot" if model.startswith("groot") else "lerobot"


def get(model, key):
    fam = family_of(model)
    r = RUNGS[fam].get(key) or EXTRA[fam].get(key)
    if r is None:
        raise SystemExit(f"unknown rung {key!r} for {model}; have {sorted(RUNGS[fam]) + sorted(EXTRA[fam])}")
    return r


def set_precision(prec):
    if prec == "highest":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def force_gemm_autotune(backends="ATEN,TRITON"):
    """Inductor's is_big_gpu() requires 68 SMs (torch/_inductor/utils.py:1676 in torch 2.10); Thor has 20, so
    mode="max-autotune" silently keeps cuBLAS for every GEMM. Override it, with max_autotune on for every compile."""
    import torch._inductor.config as IC
    import torch._inductor.utils as IU
    IU.is_big_gpu = lambda *a, **k: True
    IC.max_autotune = True
    IC.max_autotune_gemm = True
    IC.max_autotune_gemm_backends = backends
    return f"is_big_gpu forced True; max_autotune, max_autotune_gemm on; backends {backends}"


def owner(adapter, part):
    """(object, attribute) of a part's callable."""
    if adapter.family == "lerobot":
        m = adapter.policy.model
        return {"ae": (m, "denoise_step"), "vis": (getattr(m, adapter.spec.container), "embed_image"),
                "vlm": (getattr(m, adapter.spec.container), "forward"), "whole": (m, "sample_actions")}[part]
    import regions as R
    model = adapter.policy.model
    vis_owner, vis_attr, lm, _, _ = R.groot_parts(model)
    return {"ae": (model.action_head, "get_action"), "vis": (vis_owner, vis_attr), "llm": (lm, "forward")}[part]


def apply_rung(adapter, key):
    """Put the adapter's model into rung `key` (precision, preps, compilation; NOT the regions: install those after
    this returns). Returns a description dict; raises NotApplicable."""
    from graph_clone import clone_graph_outputs
    r = get(adapter.name, key)
    if r["only"] and adapter.name not in r["only"]:
        raise NotApplicable(f"rung {key} applies to {r['only']} only")
    desc = {"rung": key, "label": r["label"], "precision_flags": set_precision(r["precision"]), "preps": [],
            "compiled": [], "autotune": "", "regions": r["regions"], "note": r["note"]}
    preps = list(r["preps"])
    if adapter.name == "smolvla" and any(p in ("vis", "whole") and m == GRAPHS for p, m, _ in r["targets"]) \
            and "static_posids" not in preps:
        preps.insert(0, "static_posids")       # SmolVLA's vision cannot be captured without it (Orin finding)
    for p in preps:
        desc["preps"].append(PREPS[p](adapter.policy))
    if r["autotune"]:
        desc["autotune"] = force_gemm_autotune()
    t0 = time.perf_counter()
    for part, mode, clone in r["targets"]:
        obj, attr = owner(adapter, part)
        c = torch.compile(getattr(obj, attr), mode=mode)
        if clone:
            def _cl(*a, _c=c, **k):
                return clone_graph_outputs(_c(*a, **k))
            setattr(obj, attr, _cl)
        else:
            setattr(obj, attr, c)
        desc["compiled"].append(f"{type(obj).__name__}.{attr}@{mode}" + ("+clone" if clone else ""))
    desc["compile_wrap_s"] = time.perf_counter() - t0
    desc["ae_compiled"] = any(p in ("ae", "whole") for p, _, _ in r["targets"])
    return desc
