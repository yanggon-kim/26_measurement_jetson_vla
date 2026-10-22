"""T001 step 8 (A16): one inference per GR00T version, smoke test -- NOT a measurement.

Inputs as on the Orin (reference/orin/bench/groot_phases.py build_observation / _build_obs_n15): keys, frame counts
and state widths read from the checkpoint's own files; pixel and state values from np.random seeded 1234; 256x256
images; prompt "pick up the object". Embodiment tags as the Orin used them (reference/orin/reports/E4d_phase_balance.md):
N1.7 oxe_droid_relative_eef_relative_joint, N1.6 robocasa_panda_omron, N1.5 oxe_droid.

Per version: load from the pinned snapshot, weights compared with the checkpoint, then
  eager    : policy.get_action, 3 warm-up + 5 timed calls (wall clock, synchronised), seed 0 before each call
  compiled : the action head's DiT (action_head.model, called once per denoising step) wrapped in torch.compile
Run inside that version's venv, from outside the GR00T checkout:
  ~/venvs/groot16/bin/python bench/smoke_groot.py --version groot_n16 --out results/<run>
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as mr  # noqa: E402

TAGS = {"groot_n17": "oxe_droid_relative_eef_relative_joint", "groot_n16": "robocasa_panda_omron",
        "groot_n15": "oxe_droid"}
TEXT = "pick up the object"


def load(version, ckpt):
    if version == "groot_n15":
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy
        dc = DATA_CONFIG_MAP["oxe_droid"]
        return Gr00tPolicy(model_path=ckpt, embodiment_tag=TAGS[version], modality_config=dc.modality_config(),
                           modality_transform=dc.transform(), device="cuda")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    tag = EmbodimentTag.resolve(TAGS[version]) if hasattr(EmbodimentTag, "resolve") else EmbodimentTag(TAGS[version])
    return Gr00tPolicy(embodiment_tag=tag, model_path=ckpt, device="cuda")


def build_obs(version, policy, ckpt, hw=(256, 256)):
    np.random.seed(1234)
    h, w = hw
    if version == "groot_n15":
        mc = policy.get_modality_config()
        meta = json.loads((Path(ckpt) / "experiment_cfg" / "metadata.json").read_text())
        st = meta[getattr(policy.embodiment_tag, "value", policy.embodiment_tag)]["statistics"]["state"]
        obs = {}
        for k in mc["video"].modality_keys:
            obs[k if k.startswith("video.") else f"video.{k}"] = np.random.randint(
                0, 255, (len(mc["video"].delta_indices), h, w, 3), dtype=np.uint8)
        for k in mc["state"].modality_keys:
            obs[k if k.startswith("state.") else f"state.{k}"] = np.random.randn(
                len(mc["state"].delta_indices), len(st[k.split(".")[-1]]["min"])).astype(np.float32)
        for k in mc.get("language", mc.get("annotation")).modality_keys:
            obs[k if k.startswith("annotation.") else f"annotation.{k}"] = [TEXT]
        shapes = {k: list(v.shape) for k, v in obs.items() if hasattr(v, "shape")}
        return obs, shapes
    tag = policy.embodiment_tag.value
    mc = policy.modality_configs
    stats = json.loads((Path(ckpt) / "statistics.json").read_text())[tag]["state"]
    video = {k: np.random.randint(0, 255, (1, len(mc["video"].delta_indices), h, w, 3), dtype=np.uint8)
             for k in mc["video"].modality_keys}
    state = {k: np.random.randn(1, len(mc["state"].delta_indices), len(stats[k]["min"])).astype(np.float32)
             for k in mc["state"].modality_keys}
    obs = {"video": video, "state": state, "language": {policy.language_key: [[TEXT]]}}
    shapes = {"video": {k: list(v.shape) for k, v in video.items()},
              "state": {k: list(v.shape) for k, v in state.items()}}
    return obs, shapes


def weights_check(model, ckpt, n=400):
    import safetensors.torch as st
    files = sorted(Path(ckpt).glob("*.safetensors"))
    sd = model.state_dict()
    checked = matched = 0
    for f in files:
        ck = st.load_file(str(f))
        for k, v in ck.items():
            if k not in sd:
                continue
            checked += 1
            matched += bool(torch.equal(sd[k].detach().cpu().to(v.dtype), v))
            if checked >= n:
                break
        if checked >= n:
            break
    if checked < 100 or matched != checked:
        raise SystemExit(f"WEIGHT INTEGRITY FAILED: {matched}/{checked} tensors match {ckpt}")
    return {"checked": checked, "matched": matched, "files": [f.name for f in files], "verdict": "real weights"}


def flat_actions(a):
    if isinstance(a, tuple):
        a = a[0]
    return {k: np.asarray(v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in a.items()}


def calls(policy, obs, n_warm=3, n_timed=5):
    wall, out = [], None
    for i in range(n_warm + n_timed):
        torch.manual_seed(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.get_action(obs)
        torch.cuda.synchronize()
        if i >= n_warm:
            wall.append((time.perf_counter() - t0) * 1e3)
    return flat_actions(out), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=sorted(mr.GROOT_VERSIONS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = mr.GROOT_VERSIONS[a.version]
    res = {"model": a.version, "label": "smoke test, not a measurement", "checkpoint": v["checkpoint"],
           "checkpoint_revision": v["revision"], "code_commit": v["commit"], "embodiment_tag": TAGS[a.version],
           "N": v["n_default"]}
    try:
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(v["checkpoint"], revision=v["revision"], local_files_only=True)
        t0 = time.perf_counter()
        policy = load(a.version, ckpt)
        res["load_s"] = time.perf_counter() - t0
        model = policy.model
        ah = model.action_head
        res["num_inference_timesteps"] = int(ah.num_inference_timesteps)
        res["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        res["n_params"] = sum(p.numel() for p in model.parameters())
        res["weights"] = weights_check(model, ckpt)
        obs, shapes = build_obs(a.version, policy, ckpt)
        res["obs_shapes"] = shapes
        torch.cuda.reset_peak_memory_stats()
        act_e, wall_e = calls(policy, obs)
        res["eager"] = {"ok": True, "actions": {k: list(x.shape) for k, x in act_e.items()},
                        "finite": bool(all(np.isfinite(x).all() for x in act_e.values())),
                        "wall_ms_5_calls": wall_e, "peak_mem_bytes": torch.cuda.max_memory_allocated()}
        try:
            ah.model = torch.compile(ah.model, mode="default")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            act_c, wall_c = calls(policy, obs)
            num = sum(float(((act_c[k] - act_e[k]) ** 2).sum()) for k in act_e)
            den = sum(float((act_e[k] ** 2).sum()) for k in act_e)
            res["compiled_dit"] = {"ok": True, "mode": "default", "warmup_plus_timed_s": time.perf_counter() - t0,
                                   "finite": bool(all(np.isfinite(x).all() for x in act_c.values())),
                                   "wall_ms_5_calls": wall_c, "peak_mem_bytes": torch.cuda.max_memory_allocated(),
                                   "rel_l2_diff_vs_eager_same_seed": (num / den) ** 0.5 if den else "n/a: zero"}
        except Exception as exc:
            res["compiled_dit"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:3000],
                                   "traceback": traceback.format_exc()[-4000:]}
    except BaseException as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        res["traceback"] = traceback.format_exc()[-6000:]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"smoke_{a.version}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: x for k, x in res.items() if k != "traceback"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
