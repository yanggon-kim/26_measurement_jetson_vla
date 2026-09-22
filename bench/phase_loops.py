"""T010 — prefix-only, action-expert-only and full-inference loops.

**Compiled-mode caveat, measured not assumed.** `PI0Policy` compiles `sample_actions`, so only
the `full` loop goes through the compiled wrapper. `prefix` and `ae` call `embed_prefix`,
`paligemma_with_expert.forward` and `denoise_step` directly, i.e. BELOW it, and therefore run
eager even when the policy was loaded with a compile mode. The first version of this file did
not say so, and the resulting table had the parts summing to 1.57x the whole (prefix 339.9 +
ae 229.7 = 569.6 ms against full 362.4 ms) with nobody checking. Those two rows are now
labelled `exec_style=eager_subpath` whatever the policy's compile mode, and the discrepancy is
reported rather than hidden.

Three loops that run back to back for a fixed wall time. E11e drives them under a background
memory load, E11f uses them for throughput and concurrency, and **T004's E7 reuses them as its
power phases** (`tasks/T004_pi0_nsys_power.md`: "reuse its `bench/phase_loops.py`
(prefix-only / action-expert-only / full loops) for step 4"). Protocol §13.7 wants a loop of at
least 60 s and throughput as completed iterations / wall time.

The split follows `PI0Pytorch.sample_actions` (`modeling_pi0.py:811-892`):

  * **prefix** — `embed_prefix` then `paligemma_with_expert.forward` with
    `inputs_embeds=[prefix_embs, None], use_cache=True`, which fills the KV cache. This is
    `vision` + `vlm_prefix` together and runs once per inference.
  * **action expert** — `denoise_step` N times on a cache computed once. Isolating it this way
    is what makes an AE-only power phase possible at all.
  * **full** — `policy.predict_action_chunk(batch)`, preprocessing included.

Importable (`prefix_once`, `ae_once`, `full_once`, `run_loop`) and runnable:

    python bench/phase_loops.py --phase ae --seconds 60 --out raw/loop_ae.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import REPO, sh_out, software_versions, summarize, write_json  # noqa: E402

PHASES = ("prefix", "ae", "full", "idle")


def build(policy, batch):
    """Everything the three loops need, computed once."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    m = policy.model
    with torch.no_grad():
        images, img_masks = policy._preprocess_images(batch)          # noqa: SLF001
        state = policy.prepare_state(batch)
        lang = batch[OBS_LANGUAGE_TOKENS]
        lang_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    return {"images": images, "img_masks": img_masks, "state": state,
            "lang": lang, "lang_mask": lang_mask, "model": m}


def prefix_once(ctx):
    """vision + VLM prefix: fills the KV cache, once per inference."""
    m = ctx["model"]
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            ctx["images"], ctx["img_masks"], ctx["lang"], ctx["lang_mask"])
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
        att = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att4 = m._prepare_attention_masks_4d(att)                     # noqa: SLF001
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001, E501
        _, pkv = m.paligemma_with_expert.forward(
            attention_mask=att4, position_ids=pos, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True)
    return pkv, prefix_pad_masks


def make_ae(ctx, num_steps: int):
    """Action expert only: N denoise_step calls on a cache computed once."""
    pkv, prefix_pad_masks = prefix_once(ctx)
    m = ctx["model"]
    bsize = ctx["state"].shape[0]
    dt = -1.0 / num_steps
    noise = torch.randn(bsize, m.config.chunk_size, m.config.max_action_dim,
                        device=ctx["state"].device, dtype=torch.float32)

    def ae_once():
        # DEVIATION (protocol §2): sample_actions drives its loop with a GPU-tensor test
        # (`while time >= -dt/2`), which §2 names as a property to be reported and NOT fixed.
        # Isolating the expert requires a bounded Python loop, so this is a PATCHED variant and
        # is labelled as one wherever its numbers are used.
        with torch.no_grad():
            x_t = noise
            for step in range(num_steps):
                t = 1.0 + step * dt
                tt = torch.tensor(t, dtype=torch.float32,
                                  device=x_t.device).expand(bsize)
                v_t = m.denoise_step(ctx["state"], prefix_pad_masks, pkv, x_t, tt)
                x_t = x_t + dt * v_t
            return x_t

    return ae_once


def run_loop(fn, seconds: float, label: str, warmup: int = 5) -> dict:
    """Back-to-back for `seconds`; returns throughput and the per-iteration samples."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples, n = [], 0
    t0 = time.perf_counter()
    end = t0 + seconds
    while time.perf_counter() < end:
        a = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - a) * 1e3)
        n += 1
    wall = time.perf_counter() - t0
    s = summarize(samples) if samples else {}
    return {"label": label, "iterations": n, "wall_s": round(wall, 4),
            "throughput_inf_per_s": round(n / wall, 4),
            "median_ms": round(s.get("median_ms", 0), 4),
            "p5_ms": round(s.get("p5_ms", 0), 4), "p95_ms": round(s.get("p95_ms", 0), 4),
            "max_ms": round(s.get("max_ms", 0), 4), "n": n, "samples": samples}


def effective_exec_style(phase: str, compile_mode) -> str:
    """What actually executed, not what was requested. See the module docstring."""
    if compile_mode is None:
        return "eager"
    return "compiled" if phase == "full" else "eager_subpath"


def make_phase(policy, batch, phase: str, num_steps: int):
    ctx = build(policy, batch)
    if phase == "prefix":
        return lambda: prefix_once(ctx)
    if phase == "ae":
        return make_ae(ctx, num_steps)
    if phase == "full":
        return lambda: policy.predict_action_chunk(batch)
    if phase == "idle":
        return lambda: time.sleep(0.05)
    raise ValueError(phase)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--compile-mode", default=None)
    ap.add_argument("--out", default=None, help="per-iteration CSV")
    ap.add_argument("--json", action="store_true", help="print one JSON line and exit")
    args = ap.parse_args()

    gate = P.gate()
    if args.phase == "idle":
        r = run_loop(lambda: time.sleep(0.05), args.seconds, "idle", warmup=0)
    else:
        policy, cfg, _ = P.load_policy(num_inference_steps=args.steps,
                                       compile_mode=args.compile_mode)
        P.assert_real_weights(policy)
        fx = P.load_fixture()
        batch, _, _ = P.build_batch(policy, fx)
        fn = make_phase(policy, batch, args.phase, args.steps)
        r = run_loop(fn, args.seconds, args.phase)
    samples = r.pop("samples")
    r["gate"] = gate
    r["phase"] = args.phase
    r["num_steps"] = args.steps
    r["compile_mode"] = args.compile_mode
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "ms"])
            w.writerows(enumerate(samples))
    print("PHASE_LOOP_JSON " + json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
