"""Load a LeRobot policy (pi0, pi05, smolvla) safely on the fixture. Ported from reference/orin/bench/pi0_common.py.

Correctness guards kept from the Orin:
1. compile_model / gradient_checkpointing are overridden on the config BEFORE construction (the checkpoints ship
   compile_model: true; PI0Pytorch.__init__ compiles sample_actions immediately, modeling_pi0.py:591).
2. assert_real_weights(): from_pretrained can swallow a load error and return random weights; compare sampled
   tensors with the checkpoint file and refuse on a mismatch.
3. Processors are per checkpoint (normalisation statistics, tokenizer). The gated PaliGemma tokenizer is replaced by
   the local ungated copy through the tokenizer_processor override (A13; env/README.md).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import torch

from model_registry import PALIGEMMA_TOKENIZER_SHA256, ModelSpec

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "bench" / "fixtures"


def ensure_paligemma_tokenizer(path: str) -> dict:
    """The local tokenizer dir: the ungated tokenizer.model (sha256-checked) + a minimal tokenizer_config.json.

    The config states the Gemma defaults the gated repo's tokenizer applies to this step: <bos> prepended, no <eos>,
    <pad> = id 0. bench/smoke_lerobot.py checks the resulting valid-token count against the Orin's P (533 valid).
    """
    model = os.path.join(path, "tokenizer.model")
    h = hashlib.sha256(open(model, "rb").read()).hexdigest()
    if h != PALIGEMMA_TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.model sha256 {h} != {PALIGEMMA_TOKENIZER_SHA256}")
    cfg = {"tokenizer_class": "GemmaTokenizer", "add_bos_token": True, "add_eos_token": False,
           "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
           "model_max_length": 1000000000000000019884624838656}
    p = os.path.join(path, "tokenizer_config.json")
    if not os.path.exists(p) or json.load(open(p)) != cfg:
        json.dump(cfg, open(p, "w"), indent=1)
    return {"dir": path, "tokenizer_model_sha256": h, "tokenizer_config": cfg}


def snapshot_dir(spec: ModelSpec) -> str:
    """The local snapshot of the pinned revision.

    lerobot @ f0d2b37beb PI0Policy.from_pretrained (modeling_pi0.py:1015-1023; same in pi05) passes
    `kwargs.get("revision")` to cached_file although `revision` is a named parameter, so the pin is dropped and
    `main` is looked up; offline (or with `main` moved) that fails, and the method then RETURNS A RANDOM MODEL.
    Loading from the snapshot directory of the pinned revision avoids it and guarantees the pinned bytes.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(spec.checkpoint, revision=spec.revision, local_files_only=True)


def load_policy(spec: ModelSpec, n: int | None = None, device: str = "cuda"):
    from lerobot.configs.policies import PreTrainedConfig

    cls = getattr(importlib.import_module(spec.policy_module), spec.policy_class)
    path = snapshot_dir(spec)
    cfg = PreTrainedConfig.from_pretrained(path)
    shipped = {k: getattr(cfg, k, "n/a: field absent") for k in
               ("compile_model", "compile_mode", "gradient_checkpointing", spec.n_field, "dtype", "chunk_size")}
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.device = device
    if n is not None:
        setattr(cfg, spec.n_field, n)
    policy = cls.from_pretrained(path, config=cfg)
    policy.to(device).eval()
    if n is not None:  # the inner model keeps its own config reference (Orin pi0_common.py)
        setattr(policy.config, spec.n_field, n)
        if hasattr(policy.model, "config"):
            setattr(policy.model.config, spec.n_field, n)
    return policy, shipped


def assert_real_weights(policy, spec: ModelSpec, n: int = 400) -> dict:
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(spec.checkpoint, "model.safetensors", revision=spec.revision)
    ck = st.load_file(f)
    sd = policy.state_dict()
    checked = matched = 0
    for k, v in list(ck.items())[:n]:
        kk = k if k in sd else ("model." + k if "model." + k in sd else None)
        if kk is None:
            continue
        checked += 1
        matched += bool(torch.equal(sd[kk].detach().cpu().to(v.dtype), v))
    del ck
    if checked < 100 or matched != checked:
        raise SystemExit(f"WEIGHT INTEGRITY FAILED: {matched}/{checked} tensors of {spec.checkpoint} match")
    return {"checked": checked, "matched": matched, "checkpoint_file": os.path.basename(f), "verdict": "real weights"}


def build_obs(policy):
    """The raw observation (before the checkpoint's processors): the fixture's two camera views mapped positionally
    onto the cameras this policy declares, channels-first float32 in [0, 1]; state; task."""
    import numpy as np
    from PIL import Image

    fixture = json.loads((FIXTURES / "observation.json").read_text())
    declared = list(getattr(policy.config, "image_features", {}) or {})
    obs = {}
    for i, (key, im_spec) in enumerate(fixture["images"].items()):
        arr = np.asarray(Image.open(FIXTURES / im_spec["file"]).convert("RGB"), dtype=np.float32) / 255.0
        obs[declared[i] if i < len(declared) else key] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    obs["observation.state"] = torch.tensor(fixture["state"], dtype=torch.float32)
    obs["task"] = fixture["task"]
    return obs


def build_batch(policy, spec: ModelSpec, device: str = "cuda"):
    """The real fixture (one LIBERO frame, two cameras, state, task) through the checkpoint's own processors."""
    from lerobot.policies.factory import make_pre_post_processors

    fixture = json.loads((FIXTURES / "observation.json").read_text())
    obs = build_obs(policy)
    overrides = {"device_processor": {"device": device}}
    tok = None
    if spec.tokenizer_override:
        tok = ensure_paligemma_tokenizer(spec.tokenizer_override)
        overrides["tokenizer_processor"] = {"tokenizer_name": spec.tokenizer_override}
    pre, post = make_pre_post_processors(policy.config, snapshot_dir(spec), preprocessor_overrides=overrides)
    return pre(obs), pre, post, {"image_keys": list(obs)[:len(fixture["images"])], "tokenizer": tok}


def token_counts(policy, spec: ModelSpec, batch) -> dict:
    """P (prefix) and S (suffix) as the model builds them (Orin pi0_common.token_counts, per-model forms)."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    prep = getattr(policy, "prepare_images", None) or policy._preprocess_images  # noqa: SLF001
    images, img_masks = prep(batch)
    lang, lang_mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = policy.prepare_state(batch) if hasattr(policy, "prepare_state") else None
    m = policy.model
    with torch.no_grad():
        if spec.name == "smolvla":
            pe, pp, _ = m.embed_prefix(images, img_masks, lang, lang_mask, state=state)
        else:
            pe, pp, _ = m.embed_prefix(images, img_masks, lang, lang_mask)
        noise = m.sample_noise((pe.shape[0], policy.config.chunk_size, policy.config.max_action_dim), pe.device)
        t = torch.ones(pe.shape[0], dtype=torch.float32, device=pe.device)
        suf = m.embed_suffix(state, noise, t) if spec.name == "pi0" else m.embed_suffix(noise, t)
    return {"prefix_tokens_P": int(pe.shape[1]), "prefix_tokens_valid": int(pp.sum()),
            "suffix_tokens_S": int(suf[0].shape[1]), "suffix_tokens_valid": int(suf[1].sum()),
            "n_images": len(images), "language_tokens": int(lang.shape[1]),
            "language_tokens_valid": int(lang_mask.sum()), "chunk_size": int(policy.config.chunk_size)}
