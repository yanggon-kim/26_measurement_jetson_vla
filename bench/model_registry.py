"""Per-model descriptors (ported from reference/orin/bench/model_registry.py).

Changes from the Orin file: every checkpoint revision is pinned (docs/00_context.md), and the PaliGemma tokenizer
is resolved to the local ungated copy (env/README.md, A13). Line references are to lerobot @ f0d2b37beb.

N field:   pi0 num_inference_steps (configuration_pi0.py:45), pi05 num_inference_steps, smolvla num_steps (:66),
           GR00T num_inference_timesteps (gr00t code).
Wrap points (Orin E4 guide): vision = <container>.embed_image, vlm_prefix = <container>.forward (prefill),
           ae_step = model.denoise_step; container = paligemma_with_expert (pi0, pi05) or vlm_with_expert (smolvla).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

PALIGEMMA_TOKENIZER_DIR = os.path.expanduser("~/models/paligemma_tokenizer")
PALIGEMMA_TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    revision: str
    policy_module: str
    policy_class: str
    n_field: str
    n_default: int
    container: str
    tokenizer_override: str | None = None   # local tokenizer dir replacing a gated hub tokenizer
    family: str = "lerobot"
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "pi0": ModelSpec(
        name="pi0", checkpoint="lerobot/pi0_libero_finetuned_v044",
        revision="45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
        policy_module="lerobot.policies.pi0", policy_class="PI0Policy",
        n_field="num_inference_steps", n_default=10, container="paligemma_with_expert",
        tokenizer_override=PALIGEMMA_TOKENIZER_DIR,
        notes="Orin: P = 816 processed (533 valid), empty_cameras=1, tokenizer_max_length=48."),
    "pi05": ModelSpec(
        name="pi05", checkpoint="lerobot/pi05_libero_base",
        revision="a217bfd3b14673cf2ce597e69997ab21866438dd",
        policy_module="lerobot.policies.pi05", policy_class="PI05Policy",
        n_field="num_inference_steps", n_default=10, container="paligemma_with_expert",
        tokenizer_override=PALIGEMMA_TOKENIZER_DIR,
        notes="tokenizer_max_length=200, empty_cameras=0: measure P, never copy pi0's."),
    "smolvla": ModelSpec(
        name="smolvla", checkpoint="lerobot/smolvla_base",
        revision="d9f33c94a60fb382c90dea2164c96845bd955e28",
        policy_module="lerobot.policies.smolvla.modeling_smolvla", policy_class="SmolVLAPolicy",
        n_field="num_steps", n_default=10, container="vlm_with_expert",
        notes="layer body inlined in SmolVLMWithExpertModel.forward: per-layer ranges need the container's attn fns."),
}

GROOT_VERSIONS = {
    "groot_n17": {"src": "~/src/Isaac-GR00T", "commit": "376ba890cff8c9de64d71d982772a9c36185fdd7",
                  "checkpoint": "nvidia/GR00T-N1.7-3B", "revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
                  "n_default": 4, "venv": "~/venvs/groot17"},
    "groot_n16": {"src": "~/src/Isaac-GR00T-n1d6", "commit": "9b37aa1ce69c73c6d165233fa88128283bba4508",
                  "checkpoint": "nvidia/GR00T-N1.6-3B", "revision": "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61",
                  "n_default": 4, "venv": "~/venvs/groot16"},
    "groot_n15": {"src": "~/src/Isaac-GR00T-n1d5", "commit": "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
                  "checkpoint": "nvidia/GR00T-N1.5-3B", "revision": "869830fc749c35f34771aa5209f923ac57e4564e",
                  "n_default": 4, "venv": "~/venvs/groot15"},
}


def get(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}; have {sorted(REGISTRY)} (GR00T: {sorted(GROOT_VERSIONS)})")
    return REGISTRY[name]
