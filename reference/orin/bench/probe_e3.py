"""T002 / E3 follow-ups — three things the first pass left open or got wrong.

1. **Is the +3.13 ms step-0 offset real, or my own instrumentation?** `bench/regions.py`
   puts the 18 `ae_layer[j]` event pairs on *one* denoising step, so the anomalous step was
   also the only instrumented one. Measured three ways: layers off, layers on step 0, layers
   on step 5. If the offset follows the instrumentation it is an artefact; if it stays on
   step 0 it is real.

2. **`copy.deepcopy(past_key_values)` at `modeling_pi0.py:919`**, executed once per denoising
   step, clones the whole prefix KV cache. `docs/00_context.md` assumes
   `kv_copied_bytes_per_step = 0`. Timed directly, and its size confirmed from the tensors.

3. **Per-image vision cost** at n >= 100 with warm-up, in its own result directory — the
   first attempt ran n = 50 from an un-committed inline script.

    python bench/probe_e3.py --outdir results/<date>_E3_pi0_eager_probe
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from common import REPO, sh_out, software_versions, summarize, write_json  # noqa: E402

N_STEPS = 10


def step_times(policy, batch, enable_layers: bool, layers_in_step: int,
               warmup: int, timed: int) -> list[list[float]]:
    rec = R.Recorder(enable_layers=enable_layers, layers_in_step=layers_in_step)
    undo = R.instrument(policy, rec)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        out = []
        for _ in range(timed):
            rec.reset()
            rec.start()
            with torch.inference_mode():
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            rec.stop()
            out.append(rec.step_times_ms())
        return out
    finally:
        undo()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate))
    policy, cfg, shipped = P.load_policy(num_inference_steps=N_STEPS, compile_mode=None)
    integrity = P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    tokens = P.token_counts(policy, batch)

    # ---------------------------------------------------- 1. step-0 control
    print("\n=== 1. step-0 offset: is it real or my instrumentation? ===")
    conditions = [("layers_off", False, -1), ("layers_on_step0", True, 0),
                  ("layers_on_step5", True, 5)]
    step_rows, step_summary = [], {}
    for label, en, where in conditions:
        runs = step_times(policy, batch, en, where, args.warmup, args.timed)
        per_step = [[r[i] for r in runs] for i in range(N_STEPS)]
        meds = [summarize(v)["median_ms"] for v in per_step]
        rest = [m for i, m in enumerate(meds) if i != 0]
        step_summary[label] = {"medians_ms": [round(m, 4) for m in meds],
                               "step0_minus_mean_rest_ms": round(meds[0] - sum(rest)/len(rest), 4)}
        for i, v in enumerate(per_step):
            for it, ms in enumerate(v):
                step_rows.append({"condition": label, "step": i, "iter": it, "ms": ms})
        print(f"  {label:18s} step0 {meds[0]:6.2f}  mean(rest) {sum(rest)/len(rest):6.2f}  "
              f"offset {meds[0]-sum(rest)/len(rest):+6.2f} ms   "
              f"step5 {meds[5]:6.2f}")

    # ---------------------------------------------------- 2. the per-step KV deepcopy
    print("\n=== 2. copy.deepcopy(past_key_values) per denoising step ===")
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    images, img_masks = policy._preprocess_images(batch)
    state = policy.prepare_state(batch)
    m = policy.model
    with torch.inference_mode():
        pe, ppm, pam = m.embed_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                      batch[OBS_LANGUAGE_ATTENTION_MASK])
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
        a2d = make_att_2d_masks(ppm, pam)
        pos = torch.cumsum(ppm, dim=1) - 1
        _, pkv = m.paligemma_with_expert.forward(
            attention_mask=m._prepare_attention_masks_4d(a2d), position_ids=pos,
            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
    kv_tensors, kv_bytes = 0, 0
    for layer in pkv:
        for t in layer:
            if torch.is_tensor(t):
                kv_tensors += 1
                kv_bytes += t.numel() * t.element_size()
    torch.cuda.synchronize()
    for _ in range(10):
        copy.deepcopy(pkv)
    torch.cuda.synchronize()
    dc = []
    for _ in range(args.timed):
        t0 = time.perf_counter()
        c = copy.deepcopy(pkv)
        torch.cuda.synchronize()
        dc.append((time.perf_counter() - t0) * 1e3)
        del c
    s_dc = summarize(dc)
    print(f"  KV cache: {kv_tensors} tensors, {kv_bytes/1e6:.2f} MB")
    print(f"  deepcopy: median {s_dc['median_ms']:.3f} ms  p5-p95 "
          f"{s_dc['p5_ms']:.3f}-{s_dc['p95_ms']:.3f}  n={s_dc['n']}")
    print(f"  -> per inference at N={N_STEPS}: {N_STEPS*s_dc['median_ms']:.2f} ms, "
          f"{N_STEPS*kv_bytes/1e6:.1f} MB cloned")

    # ---------------------------------------------------- 3. per-image vision, n >= 100
    print("\n=== 3. per-image vision cost (n = %d) ===" % args.timed)
    pg = m.paligemma_with_expert
    orig = pg.embed_image
    ev = []

    def timed_embed(self, image, _o=orig):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        r = _o(image)
        e.record()
        ev.append((s, e))
        return r

    pg.embed_image = types.MethodType(timed_embed, pg)
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        ev.clear()
        with torch.inference_mode():
            for _ in range(args.timed):
                policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        per = [s.elapsed_time(e) for s, e in ev]
    finally:
        pg.embed_image = orig
    labels = ["observation.images.image (base camera)",
              "observation.images.image2 (wrist camera)",
              "empty_camera_0 (synthesised, all -1, attention mask 0)"]
    by_img = [per[i::3] for i in range(3)]
    vis_rows, vis_summary = [], {}
    for i, v in enumerate(by_img):
        s = summarize(v)
        vis_summary[labels[i]] = {k: round(x, 4) for k, x in s.items()}
        for it, ms in enumerate(v):
            vis_rows.append({"image_slot": i, "label": labels[i], "iter": it, "ms": ms})
        print(f"  slot {i} {labels[i][:46]:46s} median {s['median_ms']:7.3f} ms  n={s['n']}")
    tot = sum(vis_summary[l]["median_ms"] for l in labels)
    dummy = vis_summary[labels[2]]["median_ms"]
    print(f"  sum of the three passes {tot:.2f} ms; dummy = {100*dummy/tot:.1f} % of it")

    # ---------------------------------------------------- write
    with (out / "step0_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "step", "iter", "ms"])
        w.writeheader()
        w.writerows(step_rows)
    with (out / "vision_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_slot", "label", "iter", "ms"])
        w.writeheader()
        w.writerows(vis_rows)
    with (out / "kv_deepcopy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "ms"])
        w.writerows([[i, v] for i, v in enumerate(dc)])

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E3", "task": "T002", "kind": "follow-up probes",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "pi0 (LeRobot implementation)",
        "model_repo": "https://github.com/huggingface/lerobot",
        "model_commit": "f0d2b37bebddf7e8852cb624712da0ad564601b6",
        "checkpoint": P.CHECKPOINT, "checkpoint_revision": P.REVISION,
        "weight_integrity": integrity, "precision": cfg.dtype, "compile_mode": "none (eager)",
        "num_denoise_steps": N_STEPS, "batch": 1,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "prefix_tokens": tokens["prefix_tokens_P"], "suffix_tokens": tokens["suffix_tokens_S"],
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "torch": sw.get("torch"), "cuda": sw.get("torch_cuda"),
        "transformers": __import__("transformers").__version__,
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/probe_e3.py --outdir {args.outdir} "
                   f"--warmup {args.warmup} --timed {args.timed}",
        "step0_control": step_summary,
        "kv_deepcopy": {"tensors": kv_tensors, "bytes": kv_bytes,
                        "ms": {k: round(v, 4) for k, v in s_dc.items()},
                        "source": "lerobot/policies/pi0/modeling_pi0.py:919, "
                                  "copy.deepcopy(past_key_values) inside denoise_step"},
        "vision_per_image": vis_summary,
        "throttled": False,
        "notes": ["Probes run after the main sweep to close three gaps an independent review "
                  "found: the step-0 offset was confounded with ae_layer instrumentation, "
                  "the per-step KV deepcopy was missed, and the first per-image vision "
                  "measurement used n=50 from an un-committed script."],
    })
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
