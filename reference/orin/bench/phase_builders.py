"""The prefix-only / action-expert-only split, for every LeRobot model rather than π0.

`bench/phase_loops.py` was written for T010 and hardcoded π0's decomposition: which
arguments `embed_prefix` takes, whether the attention mask is converted to 4-D, and the
positional signature of `denoise_step`. E7 needs the same three loops for π0.5 and SmolVLA,
so the model-specific parts are isolated here and the loop driver stays one piece of code.

## The three models really do share a shape, and differ in exactly three places

Read from the installed source, `sample_actions` in each modeling file:

                     embed_prefix args              container.forward         denoise_step
    pi0    :1271     (img, mask, lang, lang_mask)   4-D mask, attn=eager      (state, pad, kv, x_t, t) positional
    pi05   :1241     (img, mask, tokens, masks)     4-D mask, attn=eager      keyword, NO state
    smolvla:291      (..., state=state)             2-D mask, fill_kv_cache   keyword, NO state

π0.5 and SmolVLA carry the robot state **inside the prefix** (π0.5 as discrete state
tokens, SmolVLA through `embed_prefix(state=...)`), which is why their `denoise_step` has
no `state` argument. That is a real architectural difference, not an API accident, and it
means their prefix does slightly more work than π0's while their expert does slightly less.
Anything comparing the phase split across the three has to say so.

SmolVLA also skips `_prepare_attention_masks_4d` entirely — the method does not exist on
its model — and passes `fill_kv_cache=True` instead of relying on `past_key_values=None`.

## What is deliberately NOT changed

π0's code path is byte-for-byte what `phase_loops.py` ran for T010, so E7's π0 numbers stay
comparable with E11e's and E11f's. The `ae_once` protocol §2 deviation (a bounded Python
loop instead of `sample_actions`' GPU-tensor `while`) is unchanged and still labelled.
"""
from __future__ import annotations

import importlib

import torch


def _att_2d(spec):
    """`make_att_2d_masks` lives in each model's own modeling module, not in a shared one."""
    mod = importlib.import_module(spec.policy_module if "modeling" in spec.policy_module
                                  else spec.policy_module + ".modeling_" + spec.name)
    return mod.make_att_2d_masks


def build_ctx(policy, spec, batch) -> dict:
    """Everything the loops need, computed once, in this model's own way."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    m = policy.model
    with torch.no_grad():
        # SmolVLA renamed the method; pi0 and pi0.5 both kept `_preprocess_images`.
        prep = getattr(policy, "prepare_images", None) or policy._preprocess_images  # noqa: SLF001
        images, img_masks = prep(batch)
        # pi0.5 has no prepare_state at all — its state goes in as prefix tokens.
        state = (policy.prepare_state(batch)
                 if hasattr(policy, "prepare_state") else None)
        lang = batch[OBS_LANGUAGE_TOKENS]
        lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    return {"images": images, "img_masks": img_masks, "state": state,
            "lang": lang, "lang_mask": lang_mask, "model": m, "spec": spec,
            "policy": policy}


def prefix_once(ctx):
    """vision + VLM prefix: fills the KV cache. Runs once per inference in every model."""
    m, spec = ctx["model"], ctx["spec"]
    with torch.no_grad():
        if spec.name == "smolvla":
            prefix_embs, pad, att = m.embed_prefix(
                ctx["images"], ctx["img_masks"], ctx["lang"], ctx["lang_mask"],
                state=ctx["state"])
            att2d = _att_2d(spec)(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            _, pkv = m.vlm_with_expert.forward(
                attention_mask=att2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=ctx["policy"].config.use_cache, fill_kv_cache=True)
            return pkv, pad
        # pi0 and pi0.5: identical here, including the eager-attention assignment that
        # sample_actions itself performs (modeling_pi0.py:784, modeling_pi05.py:760).
        prefix_embs, pad, att = m.embed_prefix(
            ctx["images"], ctx["img_masks"], ctx["lang"], ctx["lang_mask"])
        att2d = _att_2d(spec)(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        att4 = m._prepare_attention_masks_4d(att2d)                   # noqa: SLF001
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001, E501
        _, pkv = m.paligemma_with_expert.forward(
            attention_mask=att4, position_ids=pos, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
        return pkv, pad


def make_ae(ctx, num_steps: int):
    """Action expert only: N `denoise_step` calls on a cache computed once."""
    pkv, pad = prefix_once(ctx)
    m, spec = ctx["model"], ctx["spec"]
    cfg = ctx["policy"].config
    bsize = pad.shape[0]
    dt = -1.0 / num_steps
    noise = torch.randn(bsize, cfg.chunk_size, cfg.max_action_dim,
                        device=pad.device, dtype=torch.float32)
    state = ctx["state"]
    has_state_arg = spec.name == "pi0"

    def ae_once():
        # DEVIATION (protocol §2): sample_actions drives its loop with a GPU-tensor test
        # (`while time >= -dt/2`), which §2 names as a property to be REPORTED and not
        # fixed. Isolating the expert requires a bounded Python loop, so this is a PATCHED
        # variant and is labelled as one wherever its numbers are used.
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

    return ae_once


def measure_ps(policy, spec, batch) -> dict:
    """P and S as THIS model builds them, with the short key names the callers use.

    Delegates to pi0_common.token_counts, which now takes the descriptor -- one
    definition of P and S, not two that could drift.
    """
    import pi0_common as _P
    tok = _P.token_counts(policy, batch, spec)
    return {"P": tok["prefix_tokens_P"], "P_valid": tok["prefix_tokens_valid"],
            "S": tok["suffix_tokens_S"], "S_valid": tok["suffix_tokens_valid"],
            "n_images": tok["n_images"], "state_in_prefix": tok["state_in_prefix"]}
