"""T001 step 8 addendum: P and S of a GR00T version as the model processes them, read with forward hooks during one
real get_action on the smoke-test input (bench/smoke_groot.py). Not a timing run.

  P_llm   : sequence length entering the backbone's language model (input_ids / inputs_embeds)
  P_dit   : sequence length of encoder_hidden_states the DiT attends to (the hand-over to the action expert)
  S       : sequence length of hidden_states inside the DiT (state + action tokens)
  vision  : input shape of the backbone's vision module (images / tiles)

~/venvs/groot16/bin/python bench/groot_tokens.py --version groot_n16 --out results/<run>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as mr  # noqa: E402
import smoke_groot as sg  # noqa: E402


def containers(model):
    return [c for c in (getattr(model.backbone, a, None) for a in ("model", "eagle_model")) if c is not None]


def first(mods_attr, model):
    for c in containers(model):
        for a in mods_attr:
            m = getattr(c, a, None)
            if m is not None:
                return m, f"backbone.{'model' if hasattr(model.backbone, 'model') else 'eagle_model'}.{a}"
    return None, "n/a: not found"


def shape_of(args, kwargs, names):
    for n in names:
        v = kwargs.get(n)
        if torch.is_tensor(v):
            return list(v.shape), n
    for v in args:
        if torch.is_tensor(v):
            return list(v.shape), "positional[0]"
    return "n/a: no tensor argument", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=sorted(mr.GROOT_VERSIONS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = mr.GROOT_VERSIONS[a.version]
    from huggingface_hub import snapshot_download
    ckpt = snapshot_download(v["checkpoint"], revision=v["revision"], local_files_only=True)
    policy = sg.load(a.version, ckpt)
    model = policy.model
    seen = {}
    lm, lm_path = first(("language_model",), model)
    vis, vis_path = first(("visual", "vision_model"), model)
    dit = model.action_head.model

    def rec(name, names):
        def hook(mod, args, kwargs):
            if name not in seen:
                seen[name] = shape_of(args, kwargs, names)
        return hook

    hs = [lm.register_forward_pre_hook(rec("llm", ["inputs_embeds", "input_ids"]), with_kwargs=True),
          vis.register_forward_pre_hook(rec("vision", ["pixel_values", "hidden_states", "x"]), with_kwargs=True),
          dit.register_forward_pre_hook(rec("dit_hidden", ["hidden_states"]), with_kwargs=True),
          dit.register_forward_pre_hook(rec("dit_encoder", ["encoder_hidden_states"]), with_kwargs=True)]
    obs, shapes = sg.build_obs(a.version, policy, ckpt)
    torch.manual_seed(0)
    with torch.no_grad():
        policy.get_action(obs)
    for h in hs:
        h.remove()
    res = {"model": a.version, "label": "token counts from one real get_action (hooks), not a timing run",
           "checkpoint": v["checkpoint"], "checkpoint_revision": v["revision"], "embodiment_tag": sg.TAGS[a.version],
           "obs_shapes": shapes, "llm_module": lm_path, "vision_module": vis_path,
           "llm_input": seen.get("llm"), "vision_input": seen.get("vision"),
           "dit_hidden_states": seen.get("dit_hidden"), "dit_encoder_hidden_states": seen.get("dit_encoder")}
    try:
        res["P_llm"] = seen["llm"][0][1]
        res["P_dit"] = seen["dit_encoder"][0][1]
        res["S"] = seen["dit_hidden"][0][1]
    except Exception as exc:
        res["token_error"] = f"{type(exc).__name__}: {exc}"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"tokens_{a.version}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
