"""Per-model descriptors, so the phase-balance routine runs on more than π0.

`bench/` grew around π0 and is π0-specific in six ways (module-level CHECKPOINT, the
PI0Policy import, π0-shaped token_counts, region_parameter_bytes walking
paligemma_with_expert, architecture literals in run_e3.floor_rows, and "pi0" in run
directory names). This is the descriptor the rest of the harness asks instead of assuming.

Everything below was read from the installed source, not guessed; the line references are
there so it can be re-checked when lerobot moves.

## The N field differs three ways

    pi0       num_inference_steps      = 10   configuration_pi0.py:45
    pi05      num_inference_steps      = 10   configuration_pi05.py:45
    smolvla   num_steps                = 10   configuration_smolvla.py:66
    groot     num_inference_timesteps  =  4   gr00t/configs/model/gr00t_n1d7.py:106

## The three wrap points the routine depends on

`bench/regions.py` wraps exactly three callables, and `bench/opt_ladder.py` compiles those
same three so that every region boundary stays OUTSIDE a compiled graph. They are uniform
across the LeRobot family once the container's attribute name is known:

    region        callable
    vision        <container>.embed_image
    vlm_prefix    <container>.forward        (prefill branch: inputs_embeds[1] is None)
    ae_step[i]    model.denoise_step

    pi0/pi05  container = model.paligemma_with_expert   (modeling_pi05.py:438,452,567)
    smolvla   container = model.vlm_with_expert         (smolvlm_with_expert.py:179,403)

## What is NOT uniform, and must not be assumed

* `embed_suffix` takes (state, noise, t) for π0 but (noisy_actions, timestep) for π0.5
  (modeling_pi05.py:684) and SmolVLA (modeling_smolvla.py:719). So `token_counts` is
  per-model.
* π0 preprocesses images with `policy._preprocess_images`; SmolVLA uses
  `policy.prepare_images` (modeling_smolvla.py:403).
* **`regions.py`'s `ae_layer[j]` records nothing for SmolVLA.**
  `SmolVLMWithExpertModel.forward` (:403) inlines the layer body — it calls
  `forward_attn_layer` (:197) or `forward_cross_attn_layer` (:274) and then applies
  `o_proj`, `post_attention_layernorm` and `mlp` inline at :459. Patching
  `layers[j].forward` yields an empty range, silently. `per_layer_ok` records this; the
  phase-balance routine does not need per-layer ranges, but anything that does must wrap
  that loop instead.
* π0.5 has `tokenizer_max_length = 200` and `empty_cameras = 0` against π0's 48 and 1, so
  π0's constant P = 816 does NOT carry over. Measure P per model.
* GR00T is a different codebase entirely (`gr00t`, not `lerobot`) with a DiT action head;
  its descriptor is filled in when its instrumentation is built, not guessed now.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    revision: str | None
    policy_module: str
    policy_class: str
    n_field: str                 # the config attribute holding the denoising-step count
    n_default: int
    container: str               # attribute on policy.model holding the VLM+expert pair
    family: str = "lerobot"
    per_layer_ok: bool = True    # does regions.py's ae_layer[j] record anything?
    # ---- E11 (T011/T012): where the ACTION EXPERT's layers live, and how to capture one.
    # `per_layer_ok` above only DOCUMENTED that SmolVLA's layer body is inlined. These
    # fields make that dispatch instead, so bench/host_segments.py can stop hardcoding
    # `paligemma_with_expert.gemma_expert.model.layers` at five separate sites.
    expert_layers_path: str = "gemma_expert.model.layers"   # relative to the container
    expert_norm_path: str = "gemma_expert.model.norm"       # the expert's final norm
    layer_capture: str = "patch_layer_forward"
    layer_kinds: tuple[str, ...] = ("decoder",)
    notes: str = ""

    @property
    def wrap_points(self) -> dict[str, tuple[str, str]]:
        """region -> (owner attribute path relative to policy, callable name).

        `model` means policy.model; `container` means policy.model.<self.container>.
        These are what regions.py instruments and what opt_ladder.py compiles.
        """
        return {"vision": ("container", "embed_image"),
                "vlm_prefix": ("container", "forward"),
                "ae_step": ("model", "denoise_step")}

    def owner(self, policy, which: str):
        m = policy.model
        return m if which == "model" else getattr(m, self.container)

    def resolve(self, policy, path: str):
        """Walk a dotted path from the CONTAINER, so a model's layout is data not code.

        `spec.resolve(policy, spec.expert_layers_path)` is the expert's layer list for
        every model in the family, which is what lets one piece of E11 code serve all of
        them. Raises with the full path rather than an AttributeError on the last
        component, because "no attribute 'model'" on its own has cost real debugging time.
        """
        obj = getattr(policy.model, self.container)
        walked = [self.container]
        for part in path.split("."):
            if part.endswith("]"):                     # e.g. layers[0]
                name, _, idx = part[:-1].partition("[")
                obj = getattr(obj, name)[int(idx)]
            else:
                if not hasattr(obj, part):
                    raise AttributeError(
                        f"{'.'.join(walked)} has no attribute {part!r} "
                        f"(resolving {path!r} for {self.name})")
                obj = getattr(obj, part)
            walked.append(part)
        return obj

    @property
    def expert_layers_attr(self) -> str:
        return self.expert_layers_path


REGISTRY: dict[str, ModelSpec] = {
    "pi0": ModelSpec(
        name="pi0",
        checkpoint="lerobot/pi0_libero_finetuned_v044",
        revision="45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
        policy_module="lerobot.policies.pi0", policy_class="PI0Policy",
        n_field="num_inference_steps", n_default=10,
        container="paligemma_with_expert",
        notes="P = 816 processed (533 valid) for every input: 3 image slots always "
              "encoded (empty_cameras=1) and the prompt padded to tokenizer_max_length=48.",
    ),
    "pi05": ModelSpec(
        name="pi05",
        checkpoint="lerobot/pi05_libero_base", revision=None,
        policy_module="lerobot.policies.pi05", policy_class="PI05Policy",
        n_field="num_inference_steps", n_default=10,
        container="paligemma_with_expert",
        notes="tokenizer_max_length=200 and empty_cameras=0, so pi0's constant P=816 does "
              "NOT carry over — measure P. Norms are adaptive (AdaRMS), unlike pi0 whose "
              "embed_suffix returns adarms_cond=None.",
    ),
    "smolvla": ModelSpec(
        name="smolvla",
        checkpoint="lerobot/smolvla_base", revision=None,
        # SmolVLA's package __init__ re-exports nothing, unlike pi0's and pi0.5's, so the
        # class has to be imported from the modeling module directly. Checked:
        #   lerobot.policies.smolvla.SmolVLAPolicy            -> AttributeError
        #   lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy -> ok
        policy_module="lerobot.policies.smolvla.modeling_smolvla",
        policy_class="SmolVLAPolicy",
        n_field="num_steps", n_default=10,
        container="vlm_with_expert", per_layer_ok=False,
        expert_layers_path="lm_expert.layers", expert_norm_path="lm_expert.norm",
        # The layer body is INLINED in SmolVLMWithExpertModel.forward (:456-470): it calls
        # forward_attn_layer (:197) or forward_cross_attn_layer (:274) and then applies
        # o_proj, post_attention_layernorm and mlp itself. `layers[j].forward` is never
        # called, so patching it captures nothing -- silently. The wrap point that DOES
        # see a layer's real arguments is those two methods on the container.
        layer_capture="patch_container_attn_fns",
        layer_kinds=("self_attn", "cross_attn"),
        notes="16 expert layers, 8 self-attention and 8 cross-attention, decided per index "
              "by attention_mode / self_attn_every_n_layers (smolvlm_with_expert.py:425-443). "
              "ae_layer[j] records NOTHING: the layer body is inlined in "
              "SmolVLMWithExpertModel.forward.",
    ),
}

# GR00T: a different codebase (gr00t, not lerobot) with a 16-block DiT action head,
# num_inference_timesteps = 4, AdaLayerNorm with scale+shift and no gate, and even/odd
# blocks alternating cross- and self-attention (dit.py:260). Its descriptor is written when
# its instrumentation is built — guessing the wrap points here would be the same mistake as
# assuming regions.py works for SmolVLA.
GROOT_VERSIONS = {
    "groot_n17": {"branch": "main", "commit": "376ba890cff8c9de64d71d982772a9c36185fdd7",
                  "checkpoint": "nvidia/GR00T-N1.7-3B", "backbone": "Cosmos-Reason2-2B",
                  "installed": True},
    "groot_n16": {"branch": "n1d6", "commit": "9b37aa1ce69c73c6d165233fa88128283bba4508",
                  "checkpoint": "nvidia/GR00T-N1.6-3B", "backbone": "Eagle",
                  "installed": False},
    "groot_n15": {"branch": "n1d5", "commit": "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
                  "checkpoint": "nvidia/GR00T-N1.5-3B", "backbone": "Eagle",
                  "installed": False},
}


def get(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise SystemExit(f"unknown model {name!r}; have {sorted(REGISTRY)} "
                         f"(GR00T: {sorted(GROOT_VERSIONS)}, descriptor pending)")
    return REGISTRY[name]


def load_policy(spec: ModelSpec, n: int | None = None, compile_mode: str | None = None,
                device: str = "cuda"):
    """Load a policy through its own class, with the measurement-safe config.

    Mirrors pi0_common.load_policy but reads the N field name from the descriptor, because
    it differs three ways across the family.
    """
    import importlib
    from lerobot.configs.policies import PreTrainedConfig

    mod = importlib.import_module(spec.policy_module)
    cls = getattr(mod, spec.policy_class)
    kw = {"revision": spec.revision} if spec.revision else {}
    cfg = PreTrainedConfig.from_pretrained(spec.checkpoint, **kw)
    shipped = {k: getattr(cfg, k, None) for k in
               ("compile_model", "compile_mode", "gradient_checkpointing",
                spec.n_field, "dtype")}
    cfg.compile_model = compile_mode is not None
    if compile_mode is not None:
        cfg.compile_mode = compile_mode
    cfg.gradient_checkpointing = False
    cfg.device = device
    if n is not None:
        setattr(cfg, spec.n_field, n)

    policy = cls.from_pretrained(spec.checkpoint, config=cfg, **kw)
    policy.to(device).eval()
    if n is not None:
        # the inner module keeps its own config reference and reads it when the caller
        # passes None, so both must agree or a sweep silently uses the default
        setattr(policy.config, spec.n_field, n)
        if hasattr(policy, "model") and hasattr(policy.model, "config"):
            setattr(policy.model.config, spec.n_field, n)
    return policy, cfg, shipped
