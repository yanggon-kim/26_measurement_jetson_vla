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


def prep_linear_mpad(adapter, scope="lm_expert", span=256, step=16, reps=20):
    """Pad the row count M of a Linear to the fastest M' >= M measured on this device (T005 finding,
    results/20261022_E4_prefix_pi0_vs_pi05_diag/: cuBLASLt's heuristic picks a 3.8x slower kernel for pi0's MLP gate/up
    GEMM at M = 816 than at M = 976; padding is bit-identical on the real rows because GEMM rows are independent,
    results/20261022_E4_gemm_m816_fix/).
    1. one eager inference records (M, K, N, dtype) of every nn.Linear in scope;
    2. per distinct shape, M' in {M, M+step, ..., M+span} is timed (F.linear on random operands, CUDA events, median of
       `reps`); the fastest M' is kept only if it beats M by > 5 %;
    3. those Linears get a static forward: zero-pad the rows to M', F.linear, keep the first M rows.
    scope "lm_expert": the LeRobot language model and action expert (Gemma decoders); "all": every nn.Linear."""
    import statistics
    import torch.nn.functional as F
    policy = adapter.policy
    if adapter.family == "groot":
        roots = [policy.model.backbone, policy.model.action_head]
    elif scope == "all":
        roots = [policy.model]
    elif hasattr(getattr(policy.model, adapter.spec.container), "paligemma"):
        pg = getattr(policy.model, adapter.spec.container)
        roots = [pg.paligemma.model.language_model, pg.gemma_expert]
    else:
        pg = getattr(policy.model, adapter.spec.container)
        roots = [pg.vlm.model.text_model, pg.lm_expert]
    lins = {id(m): m for r in roots for m in r.modules() if isinstance(m, torch.nn.Linear)}
    seen, hooks = {}, []
    for m in lins.values():
        hooks.append(m.register_forward_pre_hook(
            lambda mod, args: seen.setdefault(id(mod), set()).add(
                (int(args[0].numel() // args[0].shape[-1]), mod.in_features, mod.out_features, args[0].dtype))))
    adapter.iteration()
    for h in hooks:
        h.remove()

    def t(m, k, n, dt):
        x = torch.randn(m, k, device="cuda", dtype=dt)
        w = torch.randn(n, k, device="cuda", dtype=dt)
        for _ in range(3):
            F.linear(x, w)
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            F.linear(x, w)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)
    best = {}
    fixed = getattr(adapter, "mpad_fixed", None)   # replay of a recorded table (--rung best_bf16): no re-tuning
    if fixed is not None:
        for shapes in seen.values():
            for (m, k, n, dt) in shapes:
                v = fixed.get(f"M={m} K={k} N={n} {dt}")
                if v is not None:
                    best[(m, k, n, dt)] = (int(v["pad_to"]), float("nan"), float("nan"))
    for shapes in ([] if fixed is not None else seen.values()):
        for (m, k, n, dt) in shapes:
            if (m, k, n, dt) in best or m < 16:
                continue
            base = t(m, k, n, dt)
            cand = min(((t(m2, k, n, dt), m2) for m2 in range(m + step, m + span + 1, step)), default=(base, m))
            best[(m, k, n, dt)] = (cand[1], base, cand[0]) if cand[0] < 0.95 * base else (m, base, base)
    padmap = {key: v[0] for key, v in best.items() if v[0] != key[0]}
    n_wrapped = 0
    for mid, m in lins.items():
        shapes = seen.get(mid, set())
        table = {(mm, dt): padmap[(mm, kk, nn, dt)] for (mm, kk, nn, dt) in shapes if (mm, kk, nn, dt) in padmap}
        if not table:
            continue

        def fwd(x, _m=m, _table=table):
            rows = x.numel() // x.shape[-1]
            m2 = _table.get((rows, x.dtype))
            if m2 is None:
                return F.linear(x, _m.weight, _m.bias)
            x2 = F.pad(x.reshape(rows, x.shape[-1]), (0, 0, 0, m2 - rows))
            return F.linear(x2, _m.weight, _m.bias)[:rows].reshape(*x.shape[:-1], _m.out_features)
        m.forward = fwd
        n_wrapped += 1
    adapter.mpad_table = {f"M={k[0]} K={k[1]} N={k[2]} {k[3]}": {"pad_to": v[0], "ms_at_M": round(v[1], 4),
                                                                  "ms_at_pad": round(v[2], 4)} for k, v in best.items()}
    pads = {k: v for k, v in adapter.mpad_table.items() if v["pad_to"] != int(k.split()[0][2:])}
    scope = "backbone+action_head" if adapter.family == "groot" else scope
    if fixed is not None:
        scope += ", replayed from the recorded best_bf16 table"
    return f"linear_mpad({scope}): {n_wrapped} Linear modules padded; shapes padded: {pads}"


def prep_bf16_islands(policy):
    """Cast the nn.Linear layers still holding fp32 weights after LeRobot's own bf16 path to bf16 (T005 E4i, the
    "weights and activations BF16" rule). Measured on Thor after bf16_both's casts:
      pi0      13 MB   action_time_mlp_in / _out, action_in_proj, state_proj, action_out_proj
      pi0.5   475 MB   the expert's adaptive-RMSNorm `dense` Linears (466 MB, read every denoising step), time MLP, projections
      SmolVLA  13 MB   action_time_mlp, some lm_expert k_proj / v_proj, state / action projections
    They are fp32 because their inputs are created fp32 outside the model (noise, time embedding). Each gets a forward
    wrapper: the input is cast to bf16, and the output is cast back to the input's dtype, so every caller sees the
    dtype it saw before. Wraps the existing forward, so it composes with linear_mpad (run this after it)."""
    n, nbytes = 0, 0
    for name, mod in policy.model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.dtype != torch.float32:
            continue
        nbytes += sum(p.numel() * 4 for p in mod.parameters(recurse=False))
        for p in mod.parameters(recurse=False):
            p.data = p.data.to(torch.bfloat16)
        inner = mod.forward

        def fwd(x, _inner=inner):
            dt = x.dtype
            y = _inner(x.to(torch.bfloat16) if dt != torch.bfloat16 else x)
            return y.to(dt) if dt != torch.bfloat16 else y
        mod.forward = fwd
        n += 1
    if n == 0:
        raise NotApplicable("no fp32 nn.Linear left to cast")
    return f"bf16_islands: {n} fp32 Linear modules ({nbytes / 1e6:.1f} MB) -> bf16, input cast to bf16, output cast back"


PREPS = {"bf16_base": prep_bf16_base, "bf16_vision": prep_bf16_vision, "bf16_norms": prep_bf16_norms,
         "bf16_both": prep_bf16_both, "eager_bf16": prep_eager_bf16, "static_posids": prep_smolvlm_static_posids,
         "linear_mpad": prep_linear_mpad, "bf16_islands": prep_bf16_islands}
ADAPTER_PREPS = {"linear_mpad"}                 # preps that need the adapter (an input to run), not only the policy


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


def force_gemm_autotune(backends="ATEN,TRITON", global_flags=True):
    """Inductor's is_big_gpu() requires 68 SMs (torch/_inductor/utils.py:1676 in torch 2.10); Thor has 20, so
    mode="max-autotune" silently keeps cuBLAS for every GEMM. Override it, with max_autotune on for every compile."""
    import torch._inductor.config as IC
    import torch._inductor.utils as IU
    IU.is_big_gpu = lambda *a, **k: True
    IC.max_autotune_gemm_backends = backends
    if not global_flags:
        # per-part autotuning (T005 E4i stage 2): only the parts compiled with mode "max-autotune*" tune their GEMMs
        return f"is_big_gpu forced True; backends {backends}; GEMM autotuning only where mode = max-autotune*"
    IC.max_autotune = True
    IC.max_autotune_gemm = True
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
        if p.startswith("attn:"):
            import e4i_candidates as EC
            _, pre, exp = p.split(":")
            desc["preps"].append(EC.prep_attn(adapter, pre, exp))
        elif p == "no_attn_fusion":
            import e4i_candidates as EC
            desc["preps"].append(EC.prep_no_attn_fusion(adapter))
        elif p == "capture_scalar":
            import torch._dynamo.config as DC
            DC.capture_scalar_outputs = True
            desc["preps"].append("torch._dynamo.config.capture_scalar_outputs = True")
        else:
            desc["preps"].append(PREPS[p](adapter) if p in ADAPTER_PREPS else PREPS[p](adapter.policy))
    if r["autotune"] == "parts":
        desc["autotune"] = force_gemm_autotune(global_flags=False)
    elif r["autotune"]:
        desc["autotune"] = force_gemm_autotune()
    if adapter.family == "groot" and any(p == "ae" for p, _, _ in r["targets"]):
        import torch._inductor.config as IC
        IC.fallback_random = True
        desc["fallback_random"] = ("torch._inductor.config.fallback_random = True: the compiled action head draws its "
                                   "seeded noise from aten's RNG, as eager does (equivalence check)")
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


# ------------------------------------------------------------------------------------------------ E4i candidates
EXTRA["lerobot"].update({r["key"]: r for r in [
    _r("eager_bf16_mpad", "eager_bf16 + Linear M padded to the fastest measured M' (lm + expert)", "lerobot", "high",
       preps=["eager_bf16", "linear_mpad"], note="labelled variant: changes no math (GEMM rows are independent)"),
    _r("bf16_both_graphs_mpad", "bf16_both_graphs + Linear M padding (lm + expert)", "lerobot", "high",
       [("ae", G, False), ("vis", G, True), ("vlm", G, True)], ["bf16_both", "linear_mpad"],
       note="labelled variant: changes no math (GEMM rows are independent)"),
]})



def _register_e4i():
    import e4i_candidates as EC
    EXTRA["lerobot"].update({r["key"]: r for r in EC.candidates()})
    EXTRA["groot"].update({r["key"]: r for r in EC.groot_candidates()})


_register_e4i()
