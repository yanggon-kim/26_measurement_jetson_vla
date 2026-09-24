"""Load LeRobot's π0 for measurement, safely. Shared by T002 (E3), T003 (E4), T004, T010.

Three things here are not conveniences, they are correctness guards:

1. **The protocol §1 gate.** Nothing is timed unless the device is at MAXN with 16 SMs and
   `jetson_clocks` applied. A latency taken in `MODE_30W` is off by ~3.4x (T001).
2. **`compile_model` / `gradient_checkpointing` are overridden before construction.** The
   checkpoint ships `compile_model: true, compile_mode: "max-autotune"`, and
   `PI0Pytorch.__init__` wraps `sample_actions` in `torch.compile` the moment it sees that —
   overriding after `from_pretrained` is too late. T002 must measure eager.
3. **Weight integrity.** `PI0Policy.from_pretrained` catches load failures, prints a warning
   and returns a model with *randomly initialised* weights. That model still runs and still
   produces plausible latencies. `assert_real_weights()` compares sampled tensors against the
   checkpoint file and refuses to continue on a mismatch.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, sh_out  # noqa: E402

CHECKPOINT = "lerobot/pi0_libero_finetuned_v044"
REVISION = "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f"
EXPECTED_SMS = 16
GPU_CUR_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"


# --------------------------------------------------------------------------- gate

def gate(require_maxn: bool = True) -> dict:
    """Protocol §1. Returns the device state to put in meta.json; raises if not measurable."""
    mode = sh_out("nvpmodel -q | head -1")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    cur = int(Path(GPU_CUR_FREQ).read_text().strip())
    mn = int(Path(GPU_CUR_FREQ.replace("cur_freq", "min_freq")).read_text().strip())
    mx = int(Path(GPU_CUR_FREQ.replace("cur_freq", "max_freq")).read_text().strip())
    state = {
        "power_mode": mode,
        "sm_count": sms,
        "gpu_cur_freq_hz": cur,
        "gpu_min_freq_hz": mn,
        "gpu_max_freq_hz": mx,
        "jetson_clocks_locked": mn == mx,
        "tpc_pg_mask": sh_out("cat /sys/devices/platform/gpu.0/tpc_pg_mask"),
        # the dispatch-bound conclusions depend on the CPU not being the limiter, so record it
        "cpu_online": sh_out("cat /sys/devices/system/cpu/online"),
        "cpu_governor": sh_out(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_cur_freq_khz": sh_out(
            "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | sort -u | tr '\n' ' '"),
        "cpu_max_freq_khz": sh_out(
            "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort -u | tr '\n' ' '"),
    }
    if require_maxn:
        if "MAXN" not in mode:
            raise SystemExit(f"GATE FAILED (protocol §1): power mode is {mode!r}, need MAXN. "
                             f"The human sets it; see STATUS.md.")
        if sms != EXPECTED_SMS:
            raise SystemExit(f"GATE FAILED: {sms} SMs, expected {EXPECTED_SMS}. "
                             f"TPCs are power-gated — not MAXN.")
        if mn != mx:
            raise SystemExit(f"GATE FAILED: GPU clock not locked ({mn}..{mx} Hz). "
                             f"Run `sudo jetson_clocks` first.")
    return state


# --------------------------------------------------------------------------- model

@dataclass
class Loaded:
    policy: object
    preprocess: object
    postprocess: object
    batch: dict
    fixture: dict
    config: object


def load_policy(num_inference_steps: int | None = None, compile_mode: str | None = None,
                device: str = "cuda"):
    """Load π0 with the measurement-safe config. `compile_mode=None` means eager (T002)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0 import PI0Policy

    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT, revision=REVISION)
    shipped = {"compile_model": cfg.compile_model, "compile_mode": cfg.compile_mode,
               "gradient_checkpointing": cfg.gradient_checkpointing,
               "num_inference_steps": cfg.num_inference_steps, "dtype": cfg.dtype}
    # see the module docstring, guard 2
    cfg.compile_model = compile_mode is not None
    if compile_mode is not None:
        cfg.compile_mode = compile_mode
    cfg.gradient_checkpointing = False
    cfg.device = device
    if num_inference_steps is not None:
        cfg.num_inference_steps = num_inference_steps

    policy = PI0Policy.from_pretrained(CHECKPOINT, revision=REVISION, config=cfg)
    policy.to(device).eval()
    # PI0Pytorch keeps its own reference to the config; sample_actions reads it when
    # num_steps is None, so both must agree or the sweep silently uses the default.
    if num_inference_steps is not None:
        policy.config.num_inference_steps = num_inference_steps
        policy.model.config.num_inference_steps = num_inference_steps
    return policy, cfg, shipped


def assert_real_weights(policy, n: int = 400) -> dict:
    """Guard 3. `from_pretrained` can silently return random weights — prove it did not."""
    import safetensors.torch as st
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(CHECKPOINT, "model.safetensors", revision=REVISION)
    ck = st.load_file(f)
    sd = policy.state_dict()
    checked = matched = 0
    for k, v in list(ck.items())[:n]:
        kk = k if k in sd else ("model." + k if "model." + k in sd else None)
        if kk is None:
            continue
        checked += 1
        if torch.equal(sd[kk].detach().cpu().to(v.dtype), v):
            matched += 1
    if checked < 100 or matched != checked:
        raise SystemExit(
            f"WEIGHT INTEGRITY FAILED: {matched}/{checked} tensors match the checkpoint. "
            f"PI0Policy.from_pretrained swallows load errors and returns RANDOM weights — "
            f"see env/README.md §10. Refusing to measure.")
    del ck
    return {"checked": checked, "matched": matched,
            "checkpoint_file_sha256_source": "huggingface_hub cache", "verdict": "real weights"}


# --------------------------------------------------------------------------- input

def load_fixture(fixtures: Path | None = None) -> dict:
    fixtures = fixtures or (REPO / "bench" / "fixtures")
    return json.loads((fixtures / "observation.json").read_text())


def build_batch(policy, fixture: dict, fixtures: Path | None = None, device: str = "cuda",
                checkpoint: str | None = None, revision: str | None = None):
    """The observation the policy consumes, from the real fixture.

    Images go in **channels-first float32 in [0, 1]** — `PI0Policy._preprocess_images` does
    the resize-with-pad to 224x224 and the [0,1] -> [-1,1] conversion itself. `empty_camera_0`
    is deliberately NOT supplied: the policy synthesises it (a -1-filled tensor with a zero
    mask) because `config.empty_cameras == 1`, which is what training did.
    """
    import numpy as np
    from PIL import Image
    from lerobot.policies.factory import make_pre_post_processors

    fixtures = fixtures or (REPO / "bench" / "fixtures")
    # The fixture stores its two camera views under pi0's key names. Other checkpoints
    # declare their own: pi0.5 wants observation.images.{image,image2} and SmolVLA wants
    # observation.images.{camera1,camera2,camera3}. Supplying pi0's names to SmolVLA made
    # it raise "All image features are missing from the batch" — so map the fixture's
    # views positionally onto whatever this policy declares, and let the policy synthesise
    # any camera it declares but the fixture does not provide (that is what empty_cameras
    # and the missing-key path exist for).
    declared = list(getattr(policy.config, "image_features", {}) or {})
    src_keys = list(fixture["images"])
    obs = {}
    for i, (key, spec) in enumerate(fixture["images"].items()):
        im = Image.open(fixtures / spec["file"]).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0          # HWC in [0,1]
        target = declared[i] if i < len(declared) else key
        obs[target] = torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # CHW
    if declared and declared[:len(src_keys)] != src_keys:
        print(f"  fixture images remapped {src_keys} -> {declared[:len(src_keys)]}",
              flush=True)
    obs["observation.state"] = torch.tensor(fixture["state"], dtype=torch.float32)
    obs["task"] = fixture["task"]

    # The processors are per-checkpoint: they carry that model's normalisation statistics
    # and its tokenizer. Defaulting to pi0's and passing another model's policy would
    # silently normalise with the wrong statistics.
    ckpt = checkpoint or CHECKPOINT
    rev = revision if checkpoint else REVISION
    kw = {"pretrained_revision": rev} if rev else {}
    preprocess, postprocess = make_pre_post_processors(
        policy.config, ckpt,
        preprocessor_overrides={"device_processor": {"device": device}}, **kw,
    )
    batch = preprocess(obs)
    return batch, preprocess, postprocess


def token_counts(policy, batch, spec=None) -> dict:
    """P (prefix) and S (suffix) as the model actually builds them — never assumed.

    `spec` defaults to pi0's, so every caller written before T011 keeps its behaviour.
    It is needed because the three models differ in two places that this function
    touches, and getting either wrong does not raise — it returns a wrong constant:

      * SmolVLA renamed `_preprocess_images` to `prepare_images` and passes `state=` into
        `embed_prefix`, because its robot state goes in the PREFIX;
      * `embed_suffix` takes `(state, noisy_actions, timestep)` for pi0 but
        `(noisy_actions, timestep)` for pi0.5 and SmolVLA, whose state is already in the
        prefix (model_registry.py). Calling pi0's form on either raises.

    pi0's P = 816 follows from `empty_cameras=1` and `tokenizer_max_length=48`; pi0.5 has
    0 and 200. P is measured here, for whichever model was passed, and never copied.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    name = getattr(spec, "name", "pi0")
    prep = getattr(policy, "prepare_images", None) or policy._preprocess_images  # noqa: SLF001
    images, img_masks = prep(batch)
    lang = batch[OBS_LANGUAGE_TOKENS]
    lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    # pi0.5 has no prepare_state: its state is embedded as prefix tokens instead.
    state = policy.prepare_state(batch) if hasattr(policy, "prepare_state") else None
    m = policy.model
    with torch.no_grad():
        if name == "smolvla":
            prefix_embs, prefix_pad, _ = m.embed_prefix(images, img_masks, lang,
                                                        lang_mask, state=state)
        else:
            prefix_embs, prefix_pad, _ = m.embed_prefix(images, img_masks, lang, lang_mask)
        bsize = prefix_embs.shape[0]
        dev = prefix_embs.device
        noise = m.sample_noise(
            (bsize, policy.config.chunk_size, policy.config.max_action_dim), dev)
        t = torch.tensor(1.0, dtype=torch.float32, device=dev).expand(bsize)
        suffix = (m.embed_suffix(state, noise, t) if name == "pi0"
                  else m.embed_suffix(noise, t))
        suffix_embs, suffix_pad = suffix[0], suffix[1]
    return {
        "prefix_tokens_P": int(prefix_embs.shape[1]),
        "prefix_tokens_valid": int(prefix_pad.sum().item()),
        "suffix_tokens_S": int(suffix_embs.shape[1]),
        "suffix_tokens_valid": int(suffix_pad.sum().item()),
        "n_images": len(images),
        "image_shapes": [list(i.shape) for i in images],
        "image_masks": [bool(m_.all().item()) for m_ in img_masks],
        "language_tokens": int(lang.shape[1]),
        "language_tokens_valid": int(lang_mask.sum().item()),
        "state_dim_padded": int(state.shape[-1]) if state is not None else None,
        "state_in_prefix": name != "pi0",
        "chunk_size": int(policy.config.chunk_size),
    }


def region_parameter_bytes(policy) -> dict:
    """Parameters each region touches, counted from the loaded model (protocol §9)."""
    m = policy.model
    pg = m.paligemma_with_expert

    def nbytes(mod):
        return sum(p.numel() * p.element_size() for p in mod.parameters())

    vision = pg.paligemma.model.vision_tower
    projector = getattr(pg.paligemma.model, "multi_modal_projector", None)
    vlm_lm = pg.paligemma.model.language_model
    expert = pg.gemma_expert.model
    out = {
        "vision_tower": nbytes(vision),
        "multi_modal_projector": nbytes(projector) if projector is not None else 0,
        "vlm_language_model": nbytes(vlm_lm),
        "action_expert": nbytes(expert),
        "action_in_proj": nbytes(m.action_in_proj),
        "action_out_proj": nbytes(m.action_out_proj),
        "total_model": sum(p.numel() * p.element_size() for p in m.parameters()),
        "total_params": sum(p.numel() for p in m.parameters()),
    }
    for extra in ("state_proj", "action_time_mlp_in", "action_time_mlp_out",
                  "time_mlp_in", "time_mlp_out"):
        if hasattr(m, extra):
            out[extra] = nbytes(getattr(m, extra))
    return out
