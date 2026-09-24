"""T004 — what each optimisation actually reduces, one factor at a time.

The campaign has measured these optimisations in separate runs and it became possible to
attribute a gain to the wrong cause. Two concrete cases:

* E4's `step_only` compiles `denoise_step` and leaves `config.compile_model` unset, so
  TF32 is never enabled; E11f loaded the policy with `compile_mode=`, which DOES set it
  (`modeling_pi0.py:591-595` calls `torch.set_float32_matmul_precision("high")` one line
  before `torch.compile`). The prefix then looked 1.41x faster "under compile" when the
  cause was the precision flip.
* Compiling `sample_actions` (LeRobot's own path) and compiling `denoise_step` (our
  `step_only` variant) are different interventions with different scopes, and an isolated
  action-expert loop that calls `denoise_step` directly is untouched by the former.

This runs ONE ladder in which each rung differs from the one above it by exactly one
factor, and reports every region, so the attribution is read off the table instead of
argued.

**Why the compiled rungs can still be split per region.** A region boundary *inside* a
compiled callable forces a graph break, which is why a compiled `sample_actions` cannot be
instrumented at all. But `bench/regions.py` wraps three specific callables from the
outside — `paligemma_with_expert.embed_image` (`vision`), `paligemma_with_expert.forward`
(`vlm_prefix`, prefill branch) and `PI0Pytorch.denoise_step` (`ae_step[i]`). Compiling
those same callables and instrumenting afterwards puts every region boundary *outside* a
compiled graph, so the split survives. That is the trick T003's `step_only` used for the
expert; rungs 4-7 extend it to vision and the VLM prefix, which no run has done before —
T003 could only report their fixed cost falling 485.1 -> 433.6 ms (1.12x) as one number.

    0  eager, matmul_precision=highest              baseline
    1  eager, matmul_precision=high                 + TF32
    2  compile denoise_step, no CUDA graphs         + compilation   (action expert)
    3  compile denoise_step, reduce-overhead        + CUDA graphs   (action expert)
    4  ... also compile embed_image, no graphs      + compilation   (vision)
    5  ... also embed_image, reduce-overhead        + CUDA graphs   (vision)
    6  ... also compile pg.forward, no graphs       + compilation   (VLM prefix)
    7  ... also pg.forward, reduce-overhead         + CUDA graphs   (VLM prefix)
    8  compile sample_actions, reduce-overhead      LeRobot's own path, e2e only
    9  compile sample_actions, max-autotune         + autotuning,    e2e only

Rungs 0-7 are CUMULATIVE on one policy: rung k+1 adds one more compiled callable to the
same model, so each marginal speedup is attributable to that callable alone, and all of
them keep their regions. Rung 7 is therefore a fully compiled model WITH a per-region
split, which `tasks/T004` says does not exist yet. Rungs 8-9 need a fresh policy — they
compile the outer `sample_actions`, which subsumes the other three — and report
end-to-end only; compare them with rung 7's e2e, never with its per-region numbers.

    python bench/opt_ladder.py --outdir results/<date>_E4c_pi0_opt_ladder --steps 4,10
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import traceback
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR
from graph_clone import clone_graph_outputs  # noqa: E402
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

# --------------------------------------------------------------------------- Part B preps
#
# modeling_pi0.py:410-417 keeps these in fp32:
#   params_to_keep_float32 = ["vision_tower", "multi_modal_projector",
#                             "input_layernorm", "post_attention_layernorm", "model.norm"]
# with the stated reason "so we never toggle (toggle causes optimizer 'same dtype' error)".
# That is a TRAINING constraint; there is no optimizer at inference. The encoder's linear
# weights are stored BF16 in model.safetensors and upcast at load, so bf16 restores the
# stored precision rather than discarding any. Both preps change numerics and must be
# validated on LIBERO before their speedups are believed.


def prep_smolvlm_static_posids(policy):
    """Remove the one op that makes SmolVLA's vision tower impossible to CUDA-graph.

    `SmolVLMVisionEmbeddings.forward` (transformers/models/smolvlm/modeling_smolvlm.py:137)
    ends with

        position_ids[patch_attention_mask.view(B, -1)] = pos_ids[patch_attention_mask.view(B, -1)]

    a boolean-mask index_put. Its output shape depends on the mask's *values*, so it needs a
    device-to-host sync; dynamo graph-breaks there and the resulting subgraph cannot be
    stream-captured. That is the whole reason SmolVLA lost four rungs with
    "operation failed due to a previous error during capture" (traceback committed at
    results/20260922_E4c_smolvla_opt_ladder/raw/traceback_vis_graphs_N10.txt).

    In LeRobot's SmolVLA the mask is all-true BY CONSTRUCTION, not by luck:
    smolvlm_with_expert.py:180 passes `patch_attention_mask=None` unconditionally, and
    modeling_smolvlm.py:355-357 then builds it as `torch.ones(...)`. With an all-true mask
    the statement is exactly `position_ids = pos_ids`. This replacement is therefore
    numerically identical, not an approximation — but it is still declared as its own ladder
    rung rather than applied silently, so its cost is measured separately from the graphs it
    enables. Both facts are asserted here, so the rung fails loudly if either stops holding.
    """
    import transformers.models.smolvlm.modeling_smolvlm as M
    import inspect
    if type(policy).__name__ != "SmolVLAPolicy":
        raise SystemExit(f"prep_smolvlm_static_posids is SmolVLA-specific; got "
                         f"{type(policy).__name__}")
    from lerobot.policies.smolvla import smolvlm_with_expert as SWE
    src = inspect.getsource(SWE.SmolVLMWithExpertModel.embed_image)
    if "patch_attention_mask = None" not in src:
        raise SystemExit("embed_image no longer passes patch_attention_mask=None; the "
                         "all-true-mask premise of this rung must be re-checked")
    if "torch.ones" not in inspect.getsource(M.SmolVLMVisionTransformer.forward):
        raise SystemExit("SmolVLMVisionTransformer no longer builds an all-ones mask")

    def forward(self, pixel_values, patch_attention_mask):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0,
                                  1 / self.num_patches_per_side, device=pixel_values.device)
        max_patches_h = patch_attention_mask.size(1)
        max_patches_w = patch_attention_mask.size(2)
        # all-true mask => nb_patches == max_patches, so no reduction over the mask is
        # needed and the counts are static.
        step_h = 1.0 / max_patches_h
        step_w = 1.0 / max_patches_w
        h = torch.arange(max_patches_h, device=pixel_values.device,
                         dtype=torch.float32) * step_h
        w = torch.arange(max_patches_w, device=pixel_values.device,
                         dtype=torch.float32) * step_w
        h = torch.clamp(h, max=(1.0 - 1e-6)).to(pixel_values.dtype)
        w = torch.clamp(w, max=(1.0 - 1e-6)).to(pixel_values.dtype)
        bh = torch.bucketize(h, boundaries, right=True)
        bw = torch.bucketize(w, boundaries, right=True)
        pos_ids = (bh[:, None] * self.num_patches_per_side + bw[None, :]).reshape(1, -1)
        position_ids = pos_ids.expand(batch_size, -1)
        return embeddings + self.position_embedding(position_ids)

    M.SmolVLMVisionEmbeddings.forward = forward
    return ("SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the "
            "identical static position ids (mask is all-true by construction)")


class _SkipRung(Exception):
    """Internal: leave the timed block without running it."""


class NotApplicable(Exception):
    """A rung that cannot exist for this model, as distinct from one that failed.

    "SmolVLA has no fp32 layernorms to cast" is not an error and must not be reported as
    one: the row records status="not applicable" with the reason, so the cross-model table
    can say why a cell is empty instead of leaving a silent blank.
    """


def _dominant_dtype(policy):
    """The dtype most of the model's WEIGHT is in, by element count.

    Counting parameter TENSORS instead of elements gets this wrong: pi0's checkpoint is
    bfloat16 with a handful of fp32 islands, but those islands are many small norm tensors,
    so they outnumber the few huge bf16 matrices and the model reads as fp32. That is why
    pi0's bf16_base ran a cast it did not need instead of recording 'not applicable'.
    """
    from collections import Counter
    c = Counter()
    for prm in policy.model.parameters():
        c[prm.dtype] += prm.numel()
    return c.most_common(1)[0][0] if c else None


def prep_bf16_base(policy):
    """B0 - cast an fp32 checkpoint to bf16 through the model's OWN supported path.

    Part B asks what removing LeRobot's fp32 islands buys. That question only means
    something if the rest of the model is bf16, and the family splits on exactly that:

        pi0     config dtype = bfloat16  -> params_to_keep_float32 IS the fp32 island
        pi0.5   config dtype = float32   -> the keep-list is inert; nothing is bf16
        SmolVLA VLM loaded torch_dtype="bfloat16" (smolvlm_with_expert.py:80), no keep-list

    So pi0.5 needs this rung before B2/B3 can mean for it what they mean for pi0, and
    running B3 on it without this is what produced "expected mat1 and mat2 to have the same
    dtype, but got: float != c10::BFloat16" - the norms went bf16 while every Linear around
    them stayed fp32. This calls the model's own to_bfloat16_for_selected_params, so the
    dtype policy measured is LeRobot's, not one I invented.
    """
    if _dominant_dtype(policy) == torch.bfloat16:
        raise NotApplicable("checkpoint already loads bf16 (config dtype=bfloat16), so "
                            "there is no fp32 base to cast")
    pg = getattr(policy.model, "paligemma_with_expert", None)
    fn = getattr(pg, "to_bfloat16_for_selected_params", None) if pg else None
    if fn is None:
        raise NotApplicable(f"{type(policy).__name__} exposes no "
                            f"to_bfloat16_for_selected_params to cast through")
    fn("bfloat16")
    return ("paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16'): fp32 "
            "checkpoint cast to bf16 keeping LeRobot's own params_to_keep_float32")


def prep_bf16_vision(policy):
    """B2 — vision tower + projector to bf16, and stop embed_image casting up to fp32.

    pi0 and pi0.5 share this body verbatim - both cast the image up to fp32, call
    get_image_features, scale pooler_output by sqrt(hidden_size) and cast back
    (modeling_pi0.py:442-450, modeling_pi05.py:437-446) - so the same replacement is
    correct for both, and the source is checked here rather than assumed.

    SmolVLA has no fp32 vision island at all: it loads its VLM with torch_dtype="bfloat16"
    (smolvlm_with_expert.py:80) and embed_image already casts the image to the vision
    model's own dtype. For it this rung is not applicable, which is a result about LeRobot's
    dtype policy, not a failure.

    On an fp32 checkpoint the bf16 base is applied first (prep_bf16_base), because a bf16
    tower feeding fp32 Linears is the dtype mismatch, not the optimisation.
    """
    import types
    import inspect
    if type(policy).__name__ not in ("PI0Policy", "PI05Policy"):
        raise NotApplicable(
            f"{type(policy).__name__} has no fp32 vision island: LeRobot loads its vision "
            f"tower in bf16 already, so there is nothing for B2 to remove")
    base_note = ""
    if _dominant_dtype(policy) != torch.bfloat16:
        base_note = prep_bf16_base(policy) + "; "
    pg = policy.model.paligemma_with_expert
    src = inspect.getsource(type(pg).embed_image)
    for token in ("get_image_features", "pooler_output"):
        if token not in src:
            raise SystemExit(f"embed_image no longer contains {token!r}; the replacement "
                             f"body in prep_bf16_vision must be re-derived from source")
    pg.paligemma.model.vision_tower.to(torch.bfloat16)
    mmp = getattr(pg.paligemma.model, "multi_modal_projector", None)
    if mmp is not None:
        mmp.to(torch.bfloat16)

    # embed_image casts the image to fp32 and the features back; with a bf16 tower that
    # cast would force the tower to run fp32 anyway. Replace it with the same body minus
    # the cast (modeling_pi0.py:442-450).
    def embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.bfloat16:
            image = image.to(torch.bfloat16)
        feats = self.paligemma.model.get_image_features(image)
        feats = feats.pooler_output * self.paligemma.config.text_config.hidden_size ** 0.5
        return feats.to(out_dtype) if feats.dtype != out_dtype else feats

    pg.embed_image = types.MethodType(embed_image, pg)
    return base_note + ("vision_tower+multi_modal_projector->bf16, embed_image fp32 cast "
                        "removed")


def prep_bf16_norms(policy):
    """B3 - every layernorm to bf16. Reaches vlm_prefix and the expert, not just vision.

    The names come from LeRobot's own params_to_keep_float32 (modeling_pi0.py:412-418,
    modeling_pi05.py:409-415), which is identical in both. SmolVLA has no such list, so
    there is no fp32 norm to cast and the rung is not applicable. As in B2, an fp32
    checkpoint gets the bf16 base first, or the norms end up bf16 among fp32 Linears.
    """
    if type(policy).__name__ not in ("PI0Policy", "PI05Policy"):
        raise NotApplicable(
            f"{type(policy).__name__} has no params_to_keep_float32: its norms are not "
            f"held in fp32, so there is nothing for B3 to cast")
    base_note = ""
    if _dominant_dtype(policy) != torch.bfloat16:
        base_note = prep_bf16_base(policy) + "; "
    # Cast the norm's OWN parameters, with recurse=False, never a submodule inside it.
    #
    # pi0.5's input_layernorm is an adaptive RMSNorm that CONTAINS a Linear ('dense',
    # 1024 -> 3x1024) belonging to the flow-matching conditioning path. mod.to(bfloat16)
    # swept that Linear up too, and three probe runs then walked the boundary one module
    # at a time -- input_layernorm.dense, then action_in_proj, then time_mlp_in -- because
    # each of those receives a tensor created OUTSIDE the model and fp32 by construction
    # (the noise, the timestep). Chasing them was the wrong move: B3 means "the layernorms
    # are no longer held in fp32", and an embedded projection is not a layernorm.
    #
    # For pi0, whose norms are plain GemmaRMSNorm with a single direct weight and no
    # submodules, this is byte-identical to the previous mod.to(bfloat16).
    n, skipped = 0, 0
    for name, mod in policy.model.named_modules():
        # Match the LAST component of the name, not a substring of it. pi0.5's adaptive
        # RMSNorm has a child module literally called
        #     ...layers.0.input_layernorm.dense
        # whose full name CONTAINS "input_layernorm", so a substring test cast the
        # conditioning Linear as if it were a norm in its own right -- which is what kept
        # producing "float != c10::BFloat16" even after the recurse=False change.
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("input_layernorm", "post_attention_layernorm") or \
                name.endswith("model.norm"):
            own = list(mod.parameters(recurse=False))
            if not own:
                continue
            for prm in own:
                prm.data = prm.data.to(torch.bfloat16)
            n += 1
            skipped += sum(1 for _ in mod.children())

    # pi0.5's "layernorm" is not only a norm. bench/pi05_dtype_probe.py, run on the real
    # model, named the failing module exactly:
    #
    #   gemma_expert.model.layers.0.input_layernorm.dense
    #   Linear  input=torch.float32  weight=torch.bfloat16  weight_shape=(3072, 1024)
    #
    # It is an ADAPTIVE RMSNorm: a Linear(1024 -> 3*1024) inside the norm that produces the
    # scale/shift/gate from the flow-matching time embedding. Casting "the norms" casts that
    # Linear too, and its input comes from time_mlp_in/out and action_in_proj, which are
    # defined on PI05FlowMatching (modeling_pi05.py:577-581) -- OUTSIDE
    # paligemma_with_expert, so LeRobot's own to_bfloat16_for_selected_params cannot reach
    # them. That is a limit of where that method lives, not a decision about precision.
    #
    # (My stated guess before the probe was that the consumer was the expert's fused QKV.
    # It was the norm's own Linear. The probe is why this is right rather than plausible.)
    #
    # So B3 also casts the fp32 conditioning path that feeds those norms: everything on
    # policy.model that is NOT inside paligemma_with_expert.
    # ...but only the modules that FEED that Linear, which is the time embedding. Casting
    # every non-expert child was too broad: the probe's second run named action_in_proj
    # (Linear 32 -> 1024) as the next mismatch, because its input is the noise tensor
    # created outside the model and fp32 by construction. action_in_proj's own fp32 output
    # is fine - bf16_base measured 591.95 ms with it fp32, so something downstream already
    # casts the suffix embeddings to the expert's dtype. time_mlp_in/out are the only
    # modules whose output reaches input_layernorm.dense.
    cond = []
    note = base_note + f"{n} layernorm modules -> bf16 (own parameters only)"
    if skipped:
        note += (f"; {skipped} submodule(s) inside those norms left in fp32 — pi0.5's "
                 f"adaptive RMSNorm embeds a conditioning Linear that is not a norm")
    return note



# (key, label, precision, [(owner, attr, mode, clone_out)], fresh, regions_ok, prep)
#
# clone_out exists for ONE measured reason. `embed_image` is called once per camera —
# three times per inference — and a reduce-overhead CUDA graph reuses a single static
# output buffer, so camera 1's embedding is overwritten by camera 2 before the caller has
# consumed it:
#     RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a
#     subsequent run
# Cloning the graph's output makes it measurable. The clone is ~1 MB per camera and is
# counted in the rung's own time, so the number is honest about what it cost; the rung is
# labelled so nobody reads it as a free win.
# attr_owner: "model" = policy.model, "pg" = policy.model.paligemma_with_expert
NOGRAPH = "max-autotune-no-cudagraphs"
GRAPHS = "reduce-overhead"
RUNGS = [
    ("eager_fp32",   "eager (matmul_precision=highest)",          "highest", [], False, True, None),
    ("eager_tf32",   "+ TF32 (matmul_precision=high)",            "high",    [], False, True, None),
    ("ae_compile",   "+ compile denoise_step (no CUDA graphs)",   "high",
     [("model", "denoise_step", NOGRAPH, False)], True, True, None),
    ("ae_graphs",    "+ CUDA graphs on denoise_step",             "high",
     [("model", "denoise_step", GRAPHS, False)], True, True, None),
    ("vis_compile",  "+ compile embed_image (no CUDA graphs)",    "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", NOGRAPH, False)], True, True, None),
    ("vis_static_pos", "+ static vision position ids (SmolVLA only; no new graph)", "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", NOGRAPH, False)],
     True, True, prep_smolvlm_static_posids),
    ("vis_graphs",   "+ CUDA graphs on embed_image (output cloned)", "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True)], True, True, None),
    ("vlm_compile",  "+ compile pg.forward (no CUDA graphs)",     "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", NOGRAPH, False)], True, True, None),
    ("vlm_graphs",   "+ CUDA graphs on pg.forward",               "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", GRAPHS, True)], True, True, None),
    ("whole_graphs", "compile sample_actions (LeRobot's path)",   "high",
     [("model", "sample_actions", GRAPHS, False)], True, False, None),
    # CORRECTION 2026-09-22: this rung was first blamed for an OOM kill. It was not the
    # cause. A second run, with whole_autotune excluded, was OOM-killed at the SAME rung
    # index (9) on a different rung, with a near-identical signature — 31.2 vs 31.7 GB
    # resident, 115.9 vs 115.6 GB virtual. The cause is memory accumulating ACROSS rungs:
    # each loads a fresh 8.28 GiB policy, and `del policy; empty_cache()` does not free it
    # because torch.compile retains references and on Orin the GPU memory IS system
    # memory. Nine models is ~74 GiB against 61 GB of RAM. Fixed by running each rung in
    # its own process (--single-rung), so the OS reclaims on exit. Kept out of the default
    # set only because T003 showed autotuning buys nothing here (1-4 % between stock
    # modes) at 5x the compile time.
    ("whole_autotune", "+ Triton autotuning (max-autotune)",      "high",
     [("model", "sample_actions", "max-autotune", False)], True, False, None),

    # An alternative to vlm_compile that omits the embed_image CUDA graph. The rungs are
    # cumulative, so a model that cannot capture embed_image loses every rung after
    # vis_graphs: SmolVLA lost four that way ("operation failed due to a previous error
    # during capture"). This keeps the prefix-compilation measurement available for such a
    # model, and is labelled so it is never compared against vlm_compile as if identical.
    ("vlm_compile_nvg", "compile pg.forward, no embed_image graph", "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "forward", NOGRAPH, False)],
     True, True, None),

    # ---- Part B: the two phases pi0's optimisation left behind --------------------
    # Each is measured on top of the expert already being compiled+graphed, because that
    # is the state the question is asked from: with the action head at 18.9 % of the
    # frame, what moves vision and vlm_prefix?
    ("bf16_base",    "B0: fp32 checkpoint -> bf16 (LeRobot's own cast)", "high",
     [("model", "denoise_step", GRAPHS, False)], True, True, prep_bf16_base),
    ("bf16_vision",  "B2: + bf16 vision tower",                   "high",
     [("model", "denoise_step", GRAPHS, False)], True, True, prep_bf16_vision),
    ("bf16_norms",   "B3: + bf16 layernorms",                     "high",
     [("model", "denoise_step", GRAPHS, False)], True, True, prep_bf16_norms),
    # Graphs on the prefix were a net loss for pi0.5 WITHOUT bf16 (871 vs 863 ms).
    # Untested on top of bf16, where the prefix is 2.16x smaller and per-launch cost is a
    # larger fraction of it -- which is exactly when graphs start paying.
    ("bf16_both_graphs", "B2+B3 with CUDA graphs on the prefix too", "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", GRAPHS, True)], True, True,
     lambda pol: prep_bf16_vision(pol) + "; " + prep_bf16_norms(pol)),
    ("bf16_both",    "B2+B3 with everything compiled",            "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", NOGRAPH, False)], True, True,
     lambda pol: prep_bf16_vision(pol) + "; " + prep_bf16_norms(pol)),
]


# SmolVLA cannot stream-capture its vision tower until the data-dependent boolean index
# in SmolVLMVisionEmbeddings is gone (see prep_smolvlm_static_posids). Every rung at or
# after vis_static_pos therefore carries that prep FOR THIS MODEL ONLY. This used to live
# inside main(), which meant apply_rung() -- factored out for E5 and E7 -- did not do it,
# and SmolVLA's vlm_graphs died in the E7 matrix with
#     torch.AcceleratorError: operation failed due to a previous error during capture
# exactly as Part B had already documented. It belongs with the rung, not with one caller.
_NEEDS_STATIC_POSIDS = {"vis_static_pos", "vis_graphs", "vlm_compile", "vlm_graphs",
                        "whole_graphs", "whole_autotune"}


def effective_prep(spec, key: str, prep):
    """The prep a rung actually needs for this model, including the SmolVLA injection."""
    if getattr(spec, "name", "") != "smolvla" or key not in _NEEDS_STATIC_POSIDS:
        return prep
    if prep is None:
        return prep_smolvlm_static_posids
    return lambda pol: "; ".join(
        x for x in (prep_smolvlm_static_posids(pol), prep(pol)) if x)


def apply_rung(policy, spec, key: str):
    """Put `policy` into the configuration a named rung describes, and say what was done.

    The ladder applies its rungs inline inside `main()`'s loop. E7 needs the SAME
    configuration — the best rung per model, not merely a `torch.compile` mode — so the
    application is factored out here rather than reimplemented. One definition, one set of
    semantics; a rung that changes here changes for both.

    Returns (compiled_desc, prep_note, precision, compile_s, fails). Raises NotApplicable
    if the rung's prep refuses this model, which the caller must handle rather than
    silently measuring an unprepared policy.
    """
    rung = next((r for r in RUNGS if r[0] == key), None)
    if rung is None:
        raise SystemExit(f"unknown rung {key!r}; have {[r[0] for r in RUNGS]}")
    _, _, prec, targets, _, _, prep = rung
    prep = effective_prep(spec, key, prep)
    torch.set_float32_matmul_precision(prec)
    prep_note = prep(policy) if prep else ""
    compiled_desc, compile_s, fails = [], 0.0, []
    for which, attr, mode, clone_out in targets:
        ownr = owner_of(policy, which, spec)
        try:
            t0 = time.time()
            c = torch.compile(getattr(ownr, attr), mode=mode)
            if clone_out:
                def _cloned(*a, _c=c, **k):
                    return clone_graph_outputs(_c(*a, **k))
                setattr(ownr, attr, _cloned)
            else:
                setattr(ownr, attr, c)
            compile_s += time.time() - t0
            compiled_desc.append(f"{which}.{attr}@{mode}" + ("+clone" if clone_out else ""))
        except Exception as e:                                    # noqa: BLE001
            fails.append(f"{which}.{attr}@{mode}: {type(e).__name__}: {str(e)[:160]}")
    return compiled_desc, prep_note, prec, round(compile_s, 3), fails



def write_tables(out: Path, rows: list[dict], steps: list[int]) -> None:
    """Write the CSVs after EVERY rung, not once at the end.

    The first full run was OOM-killed on its last rung (max-autotune on sample_actions
    reached 31.7 GB resident on a 61 GB machine). Because the tables were written only
    after the loop, all nine completed rungs were lost with it — three hours of
    measurement, recoverable only as text in a log. Partial results are now on disk from
    the first rung onward.
    """
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:                      # rows from different mechanisms differ in columns
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out / "opt_ladder.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    eff = []
    for N in steps:
        rs = [r for r in rows if r["N"] == N]
        for prev, cur in zip(rs, rs[1:]):
            e = {"N": N, "step": f"{prev['rung']} -> {cur['rung']}", "adds": cur["label"]}
            for k in ("e2e_median", "vision", "vlm_prefix", "ae_total"):
                col = k + "_ms"
                a, b = prev.get(col), cur.get(col)
                try:
                    e[k + "_speedup"] = round(float(a) / float(b), 4) if a and b else ""
                except (TypeError, ValueError, ZeroDivisionError):
                    e[k + "_speedup"] = ""
            eff.append(e)
    if eff:
        with (out / "opt_ladder_marginal.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff[0].keys()))
            w.writeheader()
            w.writerows(eff)



def mem_state() -> dict:
    """Free memory and swap, from /proc/meminfo. Recorded per rung.

    On Orin the GPU memory IS system memory, so a rung that starts under pressure can be
    slowed by reclaim or pushed into swap without any error appearing. Two runs of this
    ladder were OOM-killed at rung 9 before rungs were given their own processes; the
    numbers happened to survive intact (two independent runs agreed to 1.81 % worst case),
    but that was luck, not design. Recording the state makes it checkable instead.
    """
    mi = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = ln.partition(":")
        mi[k] = int(v.split()[0]) if v.split() else 0
    return {"mem_available_GiB": round(mi.get("MemAvailable", 0) / 1048576, 2),
            "mem_free_GiB": round(mi.get("MemFree", 0) / 1048576, 2),
            "swap_used_GiB": round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0))
                                   / 1048576, 3)}


MIN_AVAIL_GIB = 20.0     # the policy is 8.28 GiB; compile and activations need headroom



def owner_of(policy, which: str, spec=None):
    """`model` = policy.model; `pg` = the container holding embed_image and forward.

    The container is paligemma_with_expert for pi0/pi0.5 and vlm_with_expert for SmolVLA
    (bench/model_registry.py), so it cannot be hardcoded.
    """
    m = policy.model
    if which == "model":
        return m
    if spec is not None:
        return getattr(m, spec.container)
    return m.paligemma_with_expert


def finish(out: Path, args, gate, rows, notes, steps, samples=None, verified=None,
           real=None, shipped=None):
    """Write raw samples and meta.json. Shared by the per-rung parent and --in-process."""
    samples = samples or []
    sp = out / "raw" / "latency_samples.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "rung", "iter", "ms"])
        w.writeheader()
        w.writerows(samples or [])
    orig = compress_and_record(sp)

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E4c (optimisation ladder)", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_repo": P.CHECKPOINT, "model_commit": P.REVISION,
        "model_revision": P.REVISION, "checkpoint": P.CHECKPOINT, "precision": "bfloat16",
        "compile_mode": "per rung, see opt_ladder.csv", "batch": 1,
        "num_denoise_steps": steps, "prefix_tokens": 816, "suffix_tokens": 51,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": verified or real, "shipped_config": shipped,
        "command": f"python bench/opt_ladder.py --outdir {args.outdir} "
                   f"--steps {args.steps} --warmup {args.warmup} --timed {args.timed}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "originals": [orig], "not_measured": notes, "raw_files": [],
        "notes": [
            "A fresh policy is loaded for every rung. torch.compile is not reversible, and "
            "a compiled callable left on the module would contaminate the rungs below it.",
            "Rungs 0-3 keep regions on; 4-5 cannot, because a region boundary inside a "
            "compiled sample_actions forces a graph break. Compare rung 4/5's e2e with "
            "rung 3's e2e, never with its per-region numbers.",
            "`step_compile`/`whole_compile` use max-autotune-no-cudagraphs to get "
            "compilation WITHOUT CUDA graphs; the next rung adds reduce-overhead, whose "
            "difference from it is the CUDA-graph contribution. T003 found the stock modes "
            "differ by 1-4 % otherwise, so the mode change is a graph switch, not a "
            "codegen change.",
            "TF32 is set explicitly on every rung rather than inherited from "
            "config.compile_model, so that the precision factor is separated from the "
            "compilation factor instead of moving with it.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({len(rows)} rungs)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="model key from bench/model_registry.py; supplies the N field "
                         "name, checkpoint and the container holding the wrap points")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", default="4,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--in-process", action="store_true",
                    help="run every rung in this process (the old behaviour that OOMed)")
    ap.add_argument("--single-rung", default=None,
                    help="internal: run exactly this one rung and write its row to "
                         "raw/rows/<rung>_N<N>.json, then exit. The parent spawns one "
                         "process per rung so the OS reclaims the 8.28 GiB policy on exit; "
                         "in-process `del policy` does not, and nine rungs OOM-killed this "
                         "box twice at rung 9.")
    ap.add_argument("--single-N", type=int, default=None, help="internal, with --single-rung")
    ap.add_argument("--rungs",
                    default=",".join(r[0] for r in RUNGS if r[0] != "whole_autotune"),
                    help="default excludes whole_autotune, which OOM-kills this device")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    want = [r for r in RUNGS if r[0] in args.rungs.split(",")]
    spec = MR.get(args.model)          # needed by effective_prep just below

    # The SmolVLA static-posids injection is effective_prep()'s, shared with apply_rung()
    # so that E5's and E7's rungs get it too. For every other model vis_static_pos is
    # dropped rather than run, because the prep refuses a non-SmolVLA policy by design.
    if args.model == "smolvla":
        want = [r[:6] + (effective_prep(spec, r[0], r[6]),) for r in want]
    else:
        want = [r for r in want if r[0] != "vis_static_pos"]
    steps = [int(x) for x in args.steps.split(",") if x]
    steps = steps or [spec.n_default]
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    rows, notes, samples = [], [], []
    verified = None

    if not args.single_rung and not args.in_process:
        # One process per rung. `del policy; torch.cuda.empty_cache()` does NOT free the
        # 8.28 GiB policy — torch.compile keeps references, and on Orin the GPU memory is
        # system memory — so nine rungs in one process reached 31 GB resident and the
        # kernel killed the run, twice, both times at rung 9. A process exit always frees.
        d = out / "raw" / "rows"
        d.mkdir(parents=True, exist_ok=True)
        for N in steps:
            for key, label, *_ in want:
                f = d / f"{key}_N{N}.json"
                if f.exists():
                    print(f"=== N={N} {key}: already done, skipping", flush=True)
                    continue
                cmd = [sys.executable, __file__, "--model", args.model,
                       "--outdir", str(out), "--single-rung", key,
                       "--single-N", str(N), "--warmup", str(args.warmup),
                       "--timed", str(args.timed)]
                print(f"\n=== N={N} {key}: spawning", flush=True)
                r = subprocess.run(cmd)
                # Let the kernel finish reclaiming the 8.28 GiB the child held before the
                # next rung loads its own. The exit frees it, but not instantaneously, and
                # a rung that starts while reclaim is still running measures reclaim.
                for _ in range(30):
                    if mem_state()["mem_available_GiB"] >= MIN_AVAIL_GIB:
                        break
                    time.sleep(2)
                m = mem_state()
                print(f"  settled: {m['mem_available_GiB']:.1f} GiB available, "
                      f"{m['swap_used_GiB']:.2f} GiB swap", flush=True)
                if r.returncode != 0:
                    notes.append(f"N={N} {key}: subprocess exited {r.returncode} "
                                 f"(137/-9 = OOM-killed by the kernel)")
                    print(f"  rung failed, exit {r.returncode}; continuing", flush=True)
        for f in sorted(d.glob("*.json")):
            r_ = json.loads(f.read_text())
            r_.setdefault("provenance", "own process, this run")
            rows.append(r_)
        order = [r[0] for r in RUNGS]
        rows.sort(key=lambda r: (int(r["N"]),
                                 order.index(r["rung"]) if r["rung"] in order else 999))
        write_tables(out, rows, steps)
        print(f"\nmerged {len(rows)} rung(s) into {out}/opt_ladder.csv")
        return finish(out, args, gate, rows, notes, steps)

    if args.single_rung:
        want = [r for r in want if r[0] == args.single_rung]
        steps = [args.single_N]
        if not want:
            raise SystemExit(f"unknown rung {args.single_rung!r}")

    for N in steps:
        for key, label, prec, targets, fresh, regions_ok, prep in want:
            print(f"\n=== N={N}  {key}: {label}", flush=True)
            m0 = mem_state()
            print(f"  memory before: {m0['mem_available_GiB']:.1f} GiB available, "
                  f"{m0['swap_used_GiB']:.2f} GiB swap in use", flush=True)
            if m0["mem_available_GiB"] < MIN_AVAIL_GIB:
                raise SystemExit(
                    f"refusing to measure: only {m0['mem_available_GiB']:.1f} GiB available, "
                    f"below the {MIN_AVAIL_GIB} GiB floor. A rung started under memory "
                    f"pressure produces a number that looks fine and is not comparable.")
            # Every rung that compiles anything loads a fresh policy: torch.compile is not
            # reversible, so a rung cannot be un-done to build the next one on top.
            policy, cfg, shipped = MR.load_policy(spec, n=N)
            if verified is None:
                verified = (P.assert_real_weights(policy) if args.model == "pi0" else
                            {"verdict": "not verified: assert_real_weights diffs against "
                                        "pi0's own safetensors and is pi0-specific"})
                print("weights:", json.dumps(verified), flush=True)
            real = verified
            fx = P.load_fixture()
            batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                        revision=spec.revision)
            torch.set_float32_matmul_precision(prec)
            try:
                prep_note = prep(policy) if prep else ""
                na_reason = ""
            except NotApplicable as e:
                prep_note, na_reason = "", str(e)
                print(f"  NOT APPLICABLE: {na_reason}", flush=True)
            if prep_note:
                print(f"  prep: {prep_note}", flush=True)

            compiled_desc, compile_s, fails = [], 0.0, []
            for which, attr, mode, clone_out in targets:
                ownr = owner_of(policy, which, spec)
                try:
                    t0 = time.time()
                    c = torch.compile(getattr(ownr, attr), mode=mode)
                    if clone_out:
                        def _cloned(*a, _c=c, **k):
                            return clone_graph_outputs(_c(*a, **k))
                        setattr(ownr, attr, _cloned)
                    else:
                        setattr(ownr, attr, c)
                    compile_s += time.time() - t0
                    compiled_desc.append(f"{which}.{attr}@{mode}"
                                         + ("+clone" if clone_out else ""))
                except Exception as e:                            # noqa: BLE001
                    fails.append(f"{which}.{attr}@{mode}: {type(e).__name__}: {str(e)[:160]}")
            compile_s = round(compile_s, 3) if targets else ""

            # instrument AFTER compiling, so every region boundary is OUTSIDE a compiled
            # callable and the per-region split survives (the step_only trick, generalised)
            rec = undo = None
            if regions_ok:
                rec = R.Recorder(enable_layers=False, layers_in_step=0)
                undo = R.instrument(policy, rec, spec.container)

            err = None
            if na_reason:
                st, s_, reg = {"median_ms": float("nan"), "p5_ms": float("nan"),
                               "p95_ms": float("nan"), "n": 0}, [], {}
                per_iter = []
            try:
                if na_reason:
                    raise _SkipRung
                for _ in range(args.warmup):
                    policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                # Reset the recorder EVERY iteration and take the median across them, as
                # bench/run_e3.py:90-102 does. results_ms() sums repeats of a name, so
                # starting it once and reading the total gives the sum over the whole timed
                # loop: a first version reported vision 499.9 / vlm_prefix 1171.8 /
                # ae_total 1707.0 ms against an end-to-end of 409.5 ms over 5 iterations.
                per_iter, s_ = [], []
                for _ in range(args.timed):
                    if rec is not None:
                        rec.reset()
                        rec.start()
                    t0 = time.perf_counter()
                    policy.predict_action_chunk(batch)
                    torch.cuda.synchronize()
                    s_.append((time.perf_counter() - t0) * 1e3)
                    if rec is not None:
                        rec.stop()
                        per_iter.append(rec.results_ms())
                st = summarize(s_)
                reg = {}
                if per_iter:
                    for k in {k for d in per_iter for k in d}:
                        vals = [d[k] for d in per_iter if k in d]
                        reg[k] = summarize(vals)["median_ms"]
            except _SkipRung:
                pass
            except Exception as e:                                # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:200]}"
                tb_path = out / "raw" / f"traceback_{key}_N{N}.txt"
                tb_path.parent.mkdir(parents=True, exist_ok=True)
                tb_path.write_text(traceback.format_exc())
                print(f"  traceback -> {tb_path}", flush=True)
                st, s_, reg = {"median_ms": float("nan"), "p5_ms": float("nan"),
                               "p95_ms": float("nan"), "n": 0}, [], {}
                print(f"  NOT MEASURED: {err}", flush=True)
            if undo:
                undo()

            row = {"N": N, "rung": key, "label": label, "matmul_precision": prec,
                   "status": "not applicable" if na_reason else
                             ("failed" if err else "measured"),
                   "not_applicable_reason": na_reason,
                   "compiled": ";".join(compiled_desc), "regions": "on" if regions_ok
                   else "off (compiled sample_actions cannot be instrumented)",
                   "prep": prep_note, "compile_call_s": compile_s,
                   "e2e_median_ms": "" if err else round(st["median_ms"], 4),
                   "e2e_p5_ms": "" if err else round(st["p5_ms"], 4),
                   "e2e_p95_ms": "" if err else round(st["p95_ms"], 4),
                   "n": st["n"], "error": err or "",
                   "mem_available_GiB_before": m0["mem_available_GiB"],
                   "mem_available_GiB_after": mem_state()["mem_available_GiB"],
                   "swap_used_GiB_after": mem_state()["swap_used_GiB"]}
            for k in ("vision", "vlm_prefix", "ae_total"):
                row[k + "_ms"] = round(reg[k], 4) if k in reg else ""
            rows.append(row)
            samples += [{"N": N, "rung": key, "iter": i, "ms": v} for i, v in enumerate(s_)]
            if not err:
                print(f"  e2e {st['median_ms']:8.2f} ms   vision {row['vision_ms'] or '—':>10} "
                      f"vlm_prefix {row['vlm_prefix_ms'] or '—':>10} "
                      f"ae_total {row['ae_total_ms'] or '—':>10}"
                      + (f"   (compile {compile_s} s)" if targets else ""), flush=True)
            for f_ in fails:
                notes.append(f"N={N} {key}: {f_}")
                print(f"  compile failed: {f_}", flush=True)
            if err:
                notes.append(f"N={N} {key}: {err}")
            if not regions_ok:
                notes.append(f"N={N} {key}: regions off — a region boundary inside a "
                             f"compiled sample_actions forces a graph break, so only "
                             f"end-to-end is measurable for this rung.")
            if args.single_rung:
                d = out / "raw" / "rows"
                d.mkdir(parents=True, exist_ok=True)
                write_json(d / f"{key}_N{N}.json", row)
                print(f"wrote {d / f'{key}_N{N}.json'}")
                return 0
            write_tables(out, rows, steps)     # survive a kill on any later rung
            del policy
            torch.cuda.empty_cache()

    return finish(out, args, gate, rows, notes, steps, samples, verified)


if __name__ == "__main__":
    raise SystemExit(main())
