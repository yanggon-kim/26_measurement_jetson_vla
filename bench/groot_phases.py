"""GR00T phase split: backbone vs action head, at the model's own N.

GR00T cannot use bench/paper_repro.py. It lives in ~/venvs/groot (torch 2.10.0) while the
LeRobot harness lives in ~/venvs/pi0 (torch 2.8.0), and paper_repro imports lerobot. This
is the same measurement — same regions, same CSV schema, same warm-up and timing rules —
without the import.

## The two phases, read from the source (gr00t_n1d7.py:603)

    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)                    # <- vlm_prefix
    action_outputs  = self.action_head.get_action(backbone_outputs, ...) # <- ae_total

and the denoising loop is inside the action head,
`for t in range(self.num_inference_timesteps)` (:366+).

## Why vision is not split out here, stated rather than quietly omitted

In the LeRobot family `embed_image` is a distinct method, so vision separates cleanly from
the VLM. GR00T's backbone is a whole VLM invoked as one unit — Cosmos-Reason2-2B
(Qwen3-VL) for N1.7, Eagle for N1.6 and N1.5 — and isolating its vision encoder means
reaching into internals that differ across exactly the versions we are comparing. For the
action-head-share question, backbone vs action_head is the split that matters and it is
exact. `vision` is reported as "inside vlm_prefix" and the comparison table says so.

    ~/venvs/groot/bin/python bench/groot_phases.py --outdir results/<date>_E4d_groot_n17 \
        --checkpoint nvidia/GR00T-N1.7-3B
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def summarize(xs):
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[0] if n == 1 else s[int(p*(n-1))] + (
        s[min(int(p*(n-1))+1, n-1)] - s[int(p*(n-1))]) * (p*(n-1) - int(p*(n-1)))
    return {"median_ms": statistics.median(s), "mean_ms": statistics.fmean(s),
            "p5_ms": q(0.05), "p95_ms": q(0.95), "min_ms": s[0], "max_ms": s[-1], "n": n}


def gate() -> dict:
    def rd(p, d="?"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return d
    return {"gpu_cur_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"),
            "gpu_min_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq"),
            "gpu_max_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
            "cpu_online": rd("/sys/devices/system/cpu/online"),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
            "capability": str(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--version", default="n17")
    ap.add_argument("--embodiment-tag", default="oxe_droid_relative_eef_relative_joint",
                    help="must be one the checkpoint ships; it lists them on a bad value")
    ap.add_argument("--steps", type=int, default=None, help="default: the config's own")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--tf32", choices=("on", "off"), default="on")
    ap.add_argument("--head-mode", default=None,
                    help="torch.compile mode for action_head.get_action "
                         "(max-autotune-no-cudagraphs = compile only, "
                         "reduce-overhead = compile + CUDA graphs)")
    ap.add_argument("--backbone-mode", default=None,
                    help="torch.compile mode for backbone.forward")
    ap.add_argument("--rung", default="eager",
                    help="name for this configuration; one row file per rung")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    g = gate()
    print("gate:", json.dumps(g), flush=True)
    if g["gpu_cur_freq_hz"] != g["gpu_max_freq_hz"]:
        raise SystemExit(f"clocks not locked: {g}; run sudo jetson_clocks first")
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")

    from gr00t.policy.gr00t_policy import Gr00tPolicy                # noqa: E402
    from huggingface_hub import snapshot_download                     # noqa: E402

    ckpt_dir = (args.checkpoint if Path(args.checkpoint).is_dir()
                else snapshot_download(args.checkpoint))
    print(f"loading {args.checkpoint} tag={args.embodiment_tag} ...", flush=True)
    # Gr00tPolicy is the supported inference path: it loads the model in bf16 on the
    # device AND the checkpoint's own processor, so the input pipeline measured here is
    # the shipped one rather than a reimplementation.
    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=ckpt_dir,
                         device="cuda")
    model = policy.model
    n = args.steps or int(getattr(model.action_head.config, "num_inference_timesteps", 4))
    model.action_head.num_inference_timesteps = n
    print(f"N (num_inference_timesteps) = {n}", flush=True)

    # timing hooks around exactly the two phases, installed on the instance
    spans: dict[str, list] = {"vlm_prefix": [], "ae_total": []}

    def timed(obj, attr, label):
        orig = getattr(obj, attr)

        def wrapper(*a, **k):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            r = orig(*a, **k)
            e.record()
            spans[label].append((s, e))
            return r
        setattr(obj, attr, wrapper)
        return lambda: setattr(obj, attr, orig)

    compiled_desc, compile_s = [], 0.0
    if args.head_mode:
        t0 = time.time()
        model.action_head.get_action = torch.compile(model.action_head.get_action,
                                                     mode=args.head_mode)
        compile_s += time.time() - t0
        compiled_desc.append(f"action_head.get_action@{args.head_mode}")
    if args.backbone_mode:
        t0 = time.time()
        model.backbone.forward = torch.compile(model.backbone.forward,
                                               mode=args.backbone_mode)
        compile_s += time.time() - t0
        compiled_desc.append(f"backbone.forward@{args.backbone_mode}")
    if compiled_desc:
        print(f"compiled {', '.join(compiled_desc)} in {compile_s:.1f} s", flush=True)

    # NOT timed(model, "backbone", ...): that replaces the backbone MODULE with a plain
    # function, and prepare_input calls self.backbone.prepare_input (gr00t_n1d7.py:570),
    # which would raise AttributeError. Wrapping .forward leaves the module intact.
    # Instrumenting AFTER compiling keeps both region boundaries outside the compiled
    # callables, the same order opt_ladder.py uses for the LeRobot models.
    undo = [timed(model.backbone, "forward", "vlm_prefix"),
            timed(model.action_head, "get_action", "ae_total")]

    obs, shapes = build_observation(policy, ckpt_dir)
    print(f"observation shapes: {json.dumps(shapes, default=str)}", flush=True)

    # The CPU input pipeline is a real phase of GR00T inference and NVIDIA's own
    # benchmark_inference.py reports it as one, so it is timed here rather than hidden
    # by building the inputs once. The GPU phase split below then times the model alone.
    build_inputs(policy, obs)
    dp = []
    for _ in range(min(args.timed, 20)):
        t0 = time.perf_counter()
        build_inputs(policy, obs)
        dp.append((time.perf_counter() - t0) * 1e3)
    print(f"data_processing (CPU): {statistics.median(dp):.2f} ms", flush=True)

    inputs = build_inputs(policy, obs)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.get_action(inputs)
    torch.cuda.synchronize()

    e2e, per = [], {k: [] for k in spans}
    for _ in range(args.timed):
        for k in spans:
            spans[k].clear()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.get_action(inputs)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1e3)
        for k, evs in spans.items():
            per[k].append(sum(s.elapsed_time(e) for s, e in evs))
    for u in undo:
        u()

    st = summarize(e2e)
    rows = []
    for k in ("vlm_prefix", "ae_total"):
        s = summarize(per[k])
        rows.append({"region": k, **{kk: round(vv, 4) for kk, vv in s.items()},
                     "share_of_e2e": round(s["median_ms"] / st["median_ms"], 4)})
    dps = summarize(dp)
    rows.append({"region": "data_processing_cpu",
                 **{kk: round(vv, 4) for kk, vv in dps.items()},
                 "share_of_e2e": round(dps["median_ms"] / st["median_ms"], 4)})
    rows.append({"region": "e2e", **{kk: round(vv, 4) for kk, vv in st.items()},
                 "share_of_e2e": 1.0})
    rowdir = out / "raw" / "rows"
    rowdir.mkdir(parents=True, exist_ok=True)
    (rowdir / f"{args.rung}.json").write_text(json.dumps({
        "rung": args.rung, "N": n,
        "compiled": ";".join(compiled_desc), "compile_s": round(compile_s, 3),
        "e2e_median_ms": round(st["median_ms"], 4),
        "e2e_p5_ms": round(st["p5_ms"], 4), "e2e_p95_ms": round(st["p95_ms"], 4),
        "vlm_prefix_ms": round(summarize(per["vlm_prefix"])["median_ms"], 4),
        "ae_total_ms": round(summarize(per["ae_total"])["median_ms"], 4),
        "data_processing_cpu_ms": round(statistics.median(dp), 4),
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "n": st["n"],
    }, indent=1))

    with (out / "phase_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'region':12s} {'median ms':>10s} {'share':>7s}")
    for r in rows:
        print(f"{r['region']:12s} {r['median_ms']:>10.2f} {100*r['share_of_e2e']:>6.1f}%")

    (out / "meta.json").write_text(json.dumps({
        "experiment": "E4d (GR00T phase split)", "model": f"groot_{args.version}",
        "checkpoint": args.checkpoint, "precision": "bfloat16",
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "compile": ";".join(compiled_desc) or "eager",
        "rung": args.rung, "num_inference_timesteps": n,
        "n_warmup": args.warmup, "n_timed": args.timed, "gate": g,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "embodiment_tag": args.embodiment_tag,
        "observation_shapes": {k: {kk: list(vv) for kk, vv in v.items()}
                               for k, v in shapes.items()},
        "fixture_note": "keys and shapes are read from the checkpoint's own "
                        "processor_config.json and statistics.json; pixel and state "
                        "VALUES are random, which latency does not depend on",
        "data_processing_note": "data_processing_cpu is the checkpoint's own processor + "
                                "collator, timed separately; e2e below is the GPU path "
                                "only (model.get_action on already-collated inputs), the "
                                "same split NVIDIA's benchmark_inference.py reports",
        "vision_note": "GR00T's backbone is a whole VLM called as one unit, so `vision` is "
                       "INSIDE vlm_prefix here and is not separated. The LeRobot models "
                       "split it because embed_image is a distinct method.",
        "regions": {"vlm_prefix": "Gr00tN1d7.backbone (gr00t_n1d7.py:603)",
                    "ae_total": "action_head.get_action, containing the "
                                "num_inference_timesteps loop (:366+)"},
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


def build_observation(policy, ckpt_dir, image_hw=(256, 256),
                      text="pick up the object"):
    """A fixture observation in the shape THIS checkpoint declares, not a guessed one.

    Every key and every dimension is read from the checkpoint's own files:
      * which camera views, and how many frames of each -> processor_config.json's
        modality_configs[<tag>]["video"] (modality_keys, delta_indices)
      * which state vectors, and how wide each one is -> the same file for the names and
        statistics.json for the widths (len(statistics[<tag>]["state"][key]["min"]))
    Guessing these is exactly the mistake that made SmolVLA run three inferences on zero
    images without raising anything.

    The pixel and state VALUES are random. Latency does not depend on them -- there is no
    data-dependent control flow in the backbone or the flow-matching loop -- but the shapes,
    dtypes and key names do, and those are the checkpoint's. This is stated in meta.json
    rather than left for a reader to discover.
    """
    import numpy as np

    tag = policy.embodiment_tag.value
    mc = policy.modality_configs
    video_keys = mc["video"].modality_keys
    n_frames = len(mc["video"].delta_indices)
    state_keys = mc["state"].modality_keys
    n_state_steps = len(mc["state"].delta_indices)

    h, w = image_hw
    video = {k: np.random.randint(0, 255, (1, n_frames, h, w, 3), dtype=np.uint8)
             for k in video_keys}

    dims = _state_dims(ckpt_dir, tag, state_keys)
    state = {k: np.random.randn(1, n_state_steps, dims[k]).astype(np.float32)
             for k in state_keys}

    obs = {"video": video, "state": state,
           "language": {policy.language_key: [text]}}
    shapes = {"video": {k: v.shape for k, v in video.items()},
              "state": {k: v.shape for k, v in state.items()}}
    return obs, shapes


def _state_dims(ckpt_dir, tag, state_keys):
    """Width of each state vector, from the checkpoint's statistics."""
    import json
    stats = json.loads((Path(ckpt_dir) / "statistics.json").read_text())
    if tag not in stats:
        raise SystemExit(f"statistics.json has no entry for embodiment {tag!r}; "
                         f"it has {sorted(stats)}")
    out = {}
    for k in state_keys:
        if k not in stats[tag]["state"]:
            raise SystemExit(f"state key {k!r} is in the modality config but not in "
                             f"statistics.json for {tag!r}")
        out[k] = len(stats[tag]["state"][k]["min"])
    return out


def build_inputs(policy, obs):
    """Run the checkpoint's own processor + collator, exactly as inference does.

    This mirrors prepare_model_inputs in NVIDIA's own
    scripts/deployment/benchmark_inference.py: one VLAStepData per batch element through
    policy.processor, then policy.collate_fn, then everything floating-point to bf16.

    The result already contains input_ids / pixel_values, so Gr00tN1d7.prepare_input finds
    no "vlm_content" key and skips the collator: model.get_action(inputs) is then the
    GPU-only path, which is what the phase split must time. The CPU processing this
    function does is timed separately and reported as its own phase, because it is real
    inference cost that the LeRobot models' region timing also leaves out.
    """
    import torch as _t
    from gr00t.data.types import MessageType, VLAStepData

    bs = next(iter(obs["video"].values())).shape[0]
    processed = []
    for i in range(bs):
        step = VLAStepData(
            images={k: v[i] for k, v in obs["video"].items()},
            states={k: v[i] for k, v in obs["state"].items()},
            actions={},
            text=obs["language"][policy.language_key][i],
            embodiment=policy.embodiment_tag,
        )
        processed.append(policy.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": step}]))
    collated = policy.collate_fn(processed)["inputs"]

    def to_bf16(x):
        if isinstance(x, _t.Tensor) and _t.is_floating_point(x):
            return x.to(dtype=_t.bfloat16)
        if isinstance(x, dict) or hasattr(x, "items"):
            return {k: to_bf16(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_bf16(v) for v in x]
        return x

    return to_bf16(collated)


if __name__ == "__main__":
    raise SystemExit(main())
