"""Does the policy still WORK after the Part B numerics changes? Closed-loop LIBERO.

Part B's best rungs for pi0 and pi0.5 are `bf16_both`, which casts the vision tower, the
projector and the layernorms out of fp32. Those are speed results reported against an fp32
baseline, and a latency number from a model that no longer acts correctly is worthless. The
measurement plan required this check and it had not been run.

## The fp32 baseline is the harness's own control

A wrong observation mapping and a broken policy look identical from the outside: low
success rate. So every comparison runs the **fp32 baseline first, in the same harness**. If
fp32 scores near zero, the mapping is wrong and this script says the check could not be
performed — it does not report that bf16 broke the policy. Only a working fp32 baseline
makes the bf16 number meaningful.

## The observation mapping, read from the checkpoint

pi0_libero_finetuned_v044 declares:

    observation.images.image          (3, 256, 256)   <- agentview_image
    observation.images.image2         (3, 256, 256)   <- robot0_eye_in_hand_image
    observation.images.empty_camera_0 (3, 224, 224)   <- zeros, the padded third slot
    observation.state                 (8,)            <- 7 joint positions + 1 gripper
    action                            (7,)

The third camera slot is the padding T010 measured: this checkpoint always encodes three
image slots, and one of them is empty by construction.

## Resolution

With E episodes per task this resolves a policy that is broken, not a few-percent
difference in success rate. The report states the episode count and that limit.

    MUJOCO_GL=egl ~/venvs/libero/bin/python bench/libero_eval.py \
        --model pi0 --prep none --suite libero_spatial --tasks 10 --episodes 3 \
        --outdir results/<date>_E8q_pi0_libero_fp32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


SUITE_IDX = {"libero_spatial": 1, "libero_object": 2, "libero_goal": 3, "libero_10": 4}


def build_obs(obs: dict, feats: dict, task: str) -> dict:
    """LIBERO observation -> the RAW dict the checkpoint's preprocessor expects.

    Deliberately NOT the final batch. The policy needs `observation.language.tokens` and
    `observation.language.attention_mask`, which only the per-checkpoint preprocessor
    produces -- building the batch by hand gave
        KeyError: 'observation.language.tokens'
    at modeling_pi0.py:1267. The same preprocessor also applies that checkpoint's own
    normalisation statistics, so bypassing it would silently normalise with the wrong ones.
    This mirrors bench/pi0_common.py:build_batch.
    """
    out: dict = {}
    # 180 degrees, not a vertical flip. robosuite returns these images in OpenGL order and
    # the openpi LIBERO pipeline the pi0 finetune follows applies `[::-1, ::-1]`. Using
    # `[::-1]` alone is a plausible-looking image the policy was never trained on: it
    # produced sensible, in-range actions and solved 0 of 11 episodes.
    ag = obs["agentview_image"][::-1, ::-1]
    wr = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    for key, spec in feats.items():
        if not key.startswith("observation.images."):
            continue
        c, h, w = spec.shape
        if key.endswith("image"):
            img = ag
        elif key.endswith("image2"):
            img = wr
        else:
            # the padded slot this checkpoint always encodes anyway (T010)
            img = np.zeros((h, w, 3), dtype=np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if t.shape[-2:] != (h, w):
            t = torch.nn.functional.interpolate(t[None], size=(h, w), mode="bilinear",
                                                align_corners=False)[0]
        out[key] = t                                     # CHW in [0,1], unbatched
    # The 8-dim state is eef_pos(3) + axis-angle(3) + gripper_qpos(2), which is the
    # convention pi0's LIBERO finetune was trained on. My first mapping used
    # joint_pos(7) + gripper(1), also 8 wide, and it ran without error while the policy
    # solved nothing: 0/11 with every episode hitting the 280-step ceiling. Same width,
    # different meaning -- the kind of mistake that produces a complete, plausible,
    # entirely wrong accuracy number.
    from robosuite.utils.transform_utils import quat2axisangle
    st = np.concatenate([
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
    ]).astype(np.float32)
    dim = feats["observation.state"].shape[0]
    if st.shape[0] < dim:
        st = np.pad(st, (0, dim - st.shape[0]))
    out["observation.state"] = torch.from_numpy(st[:dim])
    out["task"] = task
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0")
    ap.add_argument("--prep", default="none",
                    help="none | bf16_base | bf16_vision | bf16_norms | bf16_both")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: LeRobot's own TASK_SUITE_MAX_STEPS for the suite "
                         "(lerobot/envs/libero.py: spatial 280, object 280, goal 300, "
                         "libero_10 520 -- each sized from that suite's longest training "
                         "demo). A fixed 280 would truncate every libero_10 episode.")
    ap.add_argument("--steps", type=int, default=None,
                    help="denoising steps N (T006 step 2: N in {10, 4, 2}); default = the "
                         "checkpoint's own")
    ap.add_argument("--rung", default="",
                    help="apply a full opt_ladder rung (casts AND torch.compile), e.g. "
                         "bf16_both. --prep applies only the casts; the study's speed-up is "
                         "measured on the full rung, so its accuracy must be too.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix-variant", default="pad", choices=("pad", "nocam", "trimprompt", "nopad"),
                    help="pi0 prefix padding trim (bench/pi0_padtrim.py); needs --seed-noise")
    ap.add_argument("--seed-noise", action="store_true",
                    help="T016 step 2: draw pi0's flow noise from a torch.Generator seeded per "
                         "(suite, task, episode) -- the same seed in every N arm -- and pass it to "
                         "predict_action_chunk(noise=...) instead of the unseeded global RNG")
    ap.add_argument("--num-steps-wait", type=int, default=10,
                    help="no-op steps after reset so objects settle (LeRobot's LIBERO default)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    # LIBERO's get_task_init_states() calls torch.load on its own bundled .pruned_init
    # files, which hold numpy arrays. torch >= 2.6 defaults weights_only=True and refuses
    # them:  "Unsupported global: GLOBAL numpy.core.multiarray._reconstruct".
    # These are LIBERO's own data files shipped with the benchmark, not downloaded
    # checkpoints, so the numpy reconstructors are allow-listed rather than turning
    # weights_only off globally.
    import numpy as _np
    try:
        torch.serialization.add_safe_globals([
            _np.core.multiarray._reconstruct, _np.ndarray, _np.dtype,
            _np.dtypes.Float64DType, _np.dtypes.Int64DType, _np.dtypes.UInt8DType,
        ])
    except (AttributeError, Exception):  # noqa: B014 - older numpy has no np.dtypes
        try:
            torch.serialization.add_safe_globals(
                [_np.core.multiarray._reconstruct, _np.ndarray, _np.dtype])
        except Exception:                                           # noqa: BLE001
            pass

    import model_registry as MR
    import opt_ladder as OL
    import pi0_common as P
    import pi0_padtrim as PT
    if args.prefix_variant != "pad" and not args.seed_noise:
        raise SystemExit("--prefix-variant needs --seed-noise (it bypasses select_action)")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from lerobot.envs.libero import get_libero_dummy_action

    spec = MR.get(args.model)
    policy, cfg, _ = MR.load_policy(spec, n=args.steps, compile_mode=None)
    torch.set_float32_matmul_precision("highest")
    n_effective = getattr(cfg, spec.n_field, None)
    if args.max_steps is None:
        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS
        args.max_steps = TASK_SUITE_MAX_STEPS[args.suite]
        max_steps_src = "lerobot.envs.libero.TASK_SUITE_MAX_STEPS"
    else:
        max_steps_src = "--max-steps"
    rung_note = ""
    if args.rung:
        if args.prep != "none":
            raise SystemExit("--rung already applies its own casts; do not combine with --prep")
        compiled, rprep, prec, comp_s, fails, _ = OL.apply_rung(policy, spec, args.rung)
        if fails:
            raise SystemExit(f"rung {args.rung} failed to compile: {fails}")
        rung_note = f"{args.rung}: prep=[{rprep}] compiled={compiled} matmul={prec}"
        print("rung:", rung_note, flush=True)

    prep_note = ""
    if args.prep != "none":
        fn = {"bf16_base": OL.prep_bf16_base, "bf16_vision": OL.prep_bf16_vision,
              "bf16_norms": OL.prep_bf16_norms}.get(args.prep)
        if args.prep == "bf16_both":
            prep_note = (OL.prep_bf16_vision(policy) + "; " + OL.prep_bf16_norms(policy))
        else:
            prep_note = fn(policy)
        print("prep:", prep_note, flush=True)

    # the checkpoint's own preprocessor: tokenizer + its normalisation statistics
    from lerobot.policies.factory import make_pre_post_processors
    kw = {"pretrained_revision": spec.revision} if spec.revision else {}
    preprocess, postprocess = make_pre_post_processors(
        policy.config, spec.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}}, **kw)

    feats = cfg.input_features
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[args.suite]()
    n_tasks = min(args.tasks, suite.n_tasks)

    # LeRobot's select_action keeps an internal queue: it runs the model once and then
    # returns one queued action per call. Timing every call gave a median of 3.74 ms, which
    # is the queue-hit cost, not inference -- pi0 at N=10 eager is ~1069 ms. So the calls
    # that actually ran the model are recorded separately.
    rows, t_policy, t_sim, t_infer = [], [], [], []
    # T006 raw requirement: a per-episode, PER-STEP log (policy ms and simulator ms per control
    # step). Added 2026-09-25 17:3x after the first settle10 arm (spatial fp32) was found to
    # have written none -- purely additive, nothing the rollout does changed.
    step_rows: list[dict] = []
    nonfinite_actions = torch.zeros((), dtype=torch.int64, device="cuda")  # device-side, no sync
    ep_wall_s: list[float] = []
    gate_start = P.gate()
    dtypes: dict[str, int] = {}
    for prm in policy.parameters():
        dtypes[str(prm.dtype)] = dtypes.get(str(prm.dtype), 0) + prm.numel()
    successes = total = 0
    for ti in range(n_tasks):
        task = suite.get_task(ti)
        env = OffScreenRenderEnv(bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256, camera_widths=256)
        init_states = suite.get_task_init_states(ti)
        for ep in range(args.episodes):
            env.seed(args.seed + ep)
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            # Settle before the policy acts, exactly as LeRobot's own LIBERO env does
            # (lerobot/envs/libero.py:305-308, num_steps_wait=10: "Step the simulator with a
            # no-op action for a few frames so everything settles"). The first version of this
            # harness acted from the first frame, while objects were still dropping into place:
            # libero_object scored 0/88 and spatial 18/100 against published pi0 LIBERO results
            # far higher. These steps do not count toward max_steps, as in LeRobot's env.
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            policy.reset()
            nseed = None
            if args.seed_noise:
                # seed = f(suite, task, episode) only -- independent of N, rung and run, so every arm
                # draws the same noise sequence for the same episode
                nseed = (SUITE_IDX[args.suite] * 1_000_000 + ti * 1_000 + ep) * 7919 + args.seed
                gen = torch.Generator(device="cuda").manual_seed(nseed)
                nshape = (1, policy.config.chunk_size, policy.config.max_action_dim)
                n_act = policy.config.n_action_steps
            done = False
            steps = 0
            t_ep = time.perf_counter()
            chunk: list[np.ndarray] = []
            while steps < args.max_steps and not done:
                fwd, pms = 0, 0.0
                if not chunk:
                    raw = build_obs(obs, feats, task.language)
                    batch = preprocess(raw)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        if args.seed_noise:
                            noise = torch.randn(nshape, generator=gen, device="cuda",
                                                dtype=torch.float32)
                            if args.prefix_variant == "pad":
                                act = policy.predict_action_chunk(batch, noise=noise)[:, :n_act]
                            else:
                                act = PT.predict_chunk(policy, batch, args.prefix_variant, noise)[:, :n_act]
                        else:
                            act = policy.select_action(batch)
                    nonfinite_actions += (~torch.isfinite(act)).sum()
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1e3
                    t_policy.append(dt)
                    pms = dt
                    if dt > 10.0:            # a real forward, not a queue hit
                        t_infer.append(dt)
                        fwd = 1
                    # select_action returns NORMALISED actions -- it does not
                    # unnormalise (modeling_pi0.py: it just pops the action queue). The
                    # postprocessor is what maps them back to robot units. Feeding the
                    # normalised values straight to the simulator gave 0/11 with every
                    # episode running the full 280 steps, which is what a policy doing
                    # nothing looks like.
                    act = postprocess(act)
                    if isinstance(act, dict):
                        act = act.get("action", next(iter(act.values())))
                    a = np.asarray(act).squeeze()
                    a = a.astype(np.float32)
                    chunk = [a] if a.ndim == 1 else list(a)
                t0 = time.perf_counter()
                obs, _, done, _ = env.step(chunk.pop(0))
                sms = (time.perf_counter() - t0) * 1e3
                t_sim.append(sms)
                step_rows.append({"task": ti, "episode": ep, "step": steps,
                                  "policy_call_ms": round(pms, 4), "forward": fwd,
                                  "sim_ms": round(sms, 4), "done": int(bool(done))})
                steps += 1
            ep_wall_s.append(time.perf_counter() - t_ep)
            ok = bool(done)
            successes += ok
            total += 1
            rows.append({"task": ti, "task_name": task.language, "episode": ep,
                         "success": int(ok), "steps": steps,
                         "noise_seed": nseed if nseed is not None else "unseeded (global RNG)"})
            print(f"  task {ti} ep {ep}: {'SUCCESS' if ok else 'fail':7s} "
                  f"{steps:3d} steps   running {successes}/{total}", flush=True)
        env.close()

    sr = successes / total if total else float("nan")
    med = lambda x: float(np.median(x)) if x else float("nan")
    summary = {
        "model": args.model, "prep": args.prep, "prep_note": prep_note,
        "suite": args.suite, "tasks": n_tasks, "episodes_per_task": args.episodes,
        "episodes_total": total, "successes": successes, "success_rate": round(sr, 4),
        "policy_call_ms_median": round(med(t_policy), 2),
        "policy_call_ms_note": "median over ALL select_action calls, most of which are "
                               "queue hits rather than a forward pass",
        "inference_ms_median": round(med(t_infer), 2),
        "inference_calls": len(t_infer),
        "sim_ms_median": round(med(t_sim), 2),
        "achieved_control_hz": round(1000.0 / (med(t_sim) + med(t_policy)), 2)
        if t_sim else None,
        "achieved_control_hz_note": "1000 / (median sim ms + median policy-call ms): a TYPICAL "
                                    "step, blind to the forward-pass stalls; see wall_control_hz",
        "wall_control_hz": round(len(t_sim) / sum(ep_wall_s), 3) if ep_wall_s else None,
        "wall_control_hz_note": "control steps / wall time of the policy-acting loop, summed over "
                                "episodes (settle steps and env construction excluded)",
        "forward_detect": "a select_action call > 10 ms is a forward pass; else a queue hit",
        "max_steps": args.max_steps, "max_steps_source": max_steps_src,
        "num_steps_wait": args.num_steps_wait,
        "num_steps_wait_source": "lerobot.envs.libero default (settle with no-op actions)",
        "N": n_effective, "rung": args.rung or None, "rung_note": rung_note,
        "seed": args.seed, "seeds": f"env.seed(seed + episode), episodes 0..{args.episodes-1}",
        "noise_seeding": ("torch.Generator per (suite, task, episode): seed = (suite_idx*1e6 + task*1e3 "
                          "+ episode)*7919 + seed; noise (1, chunk_size, max_action_dim) passed to "
                          "predict_action_chunk(noise=); per-episode seed in episodes.csv")
                         if args.seed_noise else "unseeded global torch RNG (select_action)",
        "nonfinite_action_elements": int(nonfinite_actions.item()),
        "prefix_variant": args.prefix_variant,
        "resolution_note": (f"{total} episodes resolves a policy that is BROKEN, not a "
                            f"few-percent difference in success rate"),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # protocol §5: meta.json per run. E8q's directories had none and needed a back-fill.
    meta = {"experiment": "E8 LIBERO closed loop (T006 step 2)", **{k: summary[k] for k in (
                "model", "prep", "rung", "suite", "N", "max_steps", "max_steps_source",
                "seed", "command", "date_utc")},
            "checkpoint": spec.checkpoint, "revision": spec.revision,
            "gate": gate_start, "gate_end": P.gate(),
            "power_mode": gate_start.get("power_mode"),
            "gpu_clock_hz": gate_start.get("gpu_cur_freq_hz"),
            "param_dtype_numel": dtypes, "matmul_precision": torch.get_float32_matmul_precision(),
            "harness_change": "2026-09-25: per-step raw log (raw/steps.csv.zst), wall_control_hz, "
                              "param dtypes and gate_end added; rollout logic unchanged",
            "torch": torch.__version__,
            "bench_commit": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import hashlib, subprocess
    sp = out / "raw" / "steps.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        w.writeheader()
        w.writerows(step_rows)
    orig = {"path": "raw/steps.csv", "bytes": sp.stat().st_size,
            "sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    subprocess.run(["zstd", "-19", "-q", "-f", "--rm", str(sp)], check=True)
    zp = sp.with_suffix(".csv.zst")
    meta["raw_files"] = [{"path": "raw/steps.csv.zst", "bytes": zp.stat().st_size,
                          "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
                          "compression": "zstd -19", "original": orig,
                          "rows": len(step_rows)}]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\n  success {successes}/{total} = {100*sr:.1f}%   "
          f"inference {summary['inference_ms_median']} ms x{summary['inference_calls']}, "
          f"sim {summary['sim_ms_median']} ms")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
