"""pi0 prefix padding trim as HARNESS-LEVEL overrides (T016 step 3 `nopad`, T017 `nocam` / `trimprompt`).

No LeRobot source is edited. The policy's own preprocessing produces the model inputs
(`policy._preprocess_images(batch)` -> images, img_masks; `policy.prepare_state(batch)`; the batch's
language tokens / mask), and the variants trim THOSE lists / tensors before they reach
`policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=...)`
(modeling_pi0.py:811) or `phase_builders.prefix_once(ctx)`:

  pad         nothing trimmed (816 prefix tokens on the fixture)
  nocam       the image slots whose mask is all False (the checkpoint's `empty_cameras: 1` slot, an all -1
              image) are REMOVED from the lists -> no SigLIP / projector call for them, no 256 tokens, no
              mask entry
  trimprompt  language tokens / mask sliced to the valid length (the processor pads to
              tokenizer_max_length = 48 on the right); refuses if the mask is not a left-aligned run of 1s
  nopad       nocam + trimprompt

Why it is exact in real arithmetic: embed_prefix gives every image token the image's mask and the
language tokens their own mask; att_masks are all 0 (bidirectional prefix); position_ids =
cumsum(pad_masks) - 1, so masked tokens do not advance the position of any later valid token; and the
2-D attention mask excludes masked keys. Removing masked tokens therefore changes no valid token's
position, mask or attention set. Bitwise equality is NOT implied (shorter GEMMs may pick other kernels).
"""
from __future__ import annotations

import torch

VARIANTS = ("pad", "nocam", "trimprompt", "nopad")


def model_inputs(policy, batch):
    """The exact tensors LeRobot's predict_action_chunk passes to sample_actions."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: PLC0415
    images, img_masks = policy._preprocess_images(batch)                                    # noqa: SLF001
    state = policy.prepare_state(batch)
    return {"images": list(images), "img_masks": list(img_masks), "state": state,
            "lang": batch[OBS_LANGUAGE_TOKENS], "lang_mask": batch[OBS_LANGUAGE_ATTENTION_MASK]}


def valid_prompt_len(lang_mask) -> int:
    m = lang_mask[0].bool()
    L = int(m.sum())
    if not bool(m[:L].all()) or bool(m[L:].any()):
        raise ValueError("language mask is not a left-aligned run of valid tokens -- refusing to trim")
    return L


def trim(inp: dict, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    out = dict(inp)
    if variant in ("nocam", "nopad"):
        keep = [i for i, mk in enumerate(inp["img_masks"]) if bool(mk.any())]
        out["images"] = [inp["images"][i] for i in keep]
        out["img_masks"] = [inp["img_masks"][i] for i in keep]
    if variant in ("trimprompt", "nopad"):
        L = valid_prompt_len(inp["lang_mask"])
        out["lang"] = inp["lang"][:, :L]
        out["lang_mask"] = inp["lang_mask"][:, :L]
    return out


def prefix_tokens(policy, inp: dict) -> dict:
    """Token counts from the tensors themselves: per-image tokens from one embed_image call's output."""
    with torch.no_grad():
        per_img = policy.model.paligemma_with_expert.embed_image(inp["images"][0]).shape[1]
    n_img = len(inp["images"])
    valid_img = sum(int(bool(mk.any())) for mk in inp["img_masks"])
    L = inp["lang"].shape[1]
    Lv = int(inp["lang_mask"][0].sum())
    return {"tokens_per_image": int(per_img), "image_slots": n_img, "valid_image_slots": valid_img,
            "prompt_positions": int(L), "valid_prompt_tokens": Lv,
            "prefix_tokens": int(n_img * per_img + L), "prefix_tokens_valid": int(valid_img * per_img + Lv)}


def sample(policy, inp: dict, noise, num_steps=None):
    kw = {"noise": noise}
    if num_steps is not None:
        kw["num_steps"] = num_steps
    return policy.model.sample_actions(inp["images"], inp["img_masks"], inp["lang"], inp["lang_mask"],
                                       inp["state"], **kw)


def ctx_for(ctx: dict, variant: str) -> dict:
    """phase_builders ctx (images, img_masks, lang, lang_mask, ...) trimmed the same way."""
    t = trim({k: ctx[k] for k in ("images", "img_masks", "lang", "lang_mask", "state")}, variant)
    return {**ctx, **t}


def predict_chunk(policy, batch, variant: str, noise):
    """policy.predict_action_chunk(batch, noise=noise) with the prefix trimmed -- the same three steps
    (preprocess, sample_actions, unpad to the action dimension), nothing else."""
    from lerobot.utils.constants import ACTION                                          # noqa: PLC0415
    inp = trim(model_inputs(policy, batch), variant)
    actions = sample(policy, inp, noise)
    return actions[:, :, :policy.config.output_features[ACTION].shape[0]]
