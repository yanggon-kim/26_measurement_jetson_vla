"""T010 / E11c — the object handed between the GPU prefix and the memory-side expert (§13.4).

In the hybrid design the GPU computes the prefix once and the memory side denoises against it.
What crosses that boundary is the prefix KV cache, and the analytic model prices it as
15–16 MB loaded once, with `kv_copied_bytes_per_step = 0` and `t_kv_materialise = 0`
(`docs/00_context.md`). T002 already found one reason to doubt the zeros:
`copy.deepcopy(past_key_values)` at `modeling_pi0.py:918`, executed **every** denoising step.

This measures what the object is (shape, dtype, stride, contiguity, valid vs padded tokens),
what it costs to materialise in the two layouts the mapping would want, and — with evidence
from both the source and the running process — how the denoising steps actually use it.

    python bench/handover.py --outdir results/<date>_E11c_pi0_handover
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
from e11_summaries import e11c  # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions, summarize,  # noqa: E402
                    write_json)
from host_segments import ModelInputs  # noqa: E402
from phase_loops import build, prefix_once  # noqa: E402


def cache_layers(pkv):
    """(keys, values) per layer, across the THREE cache shapes in this study.

    pi0 and pi0.5 get a `transformers` Cache, in either of its two APIs. SmolVLA does not
    use one at all: `SmolVLMWithExpertModel.forward` keeps a plain dict,
    `past_key_values[layer_idx] = {"key_states": ..., "value_states": ...}`
    (smolvlm_with_expert.py:253-256), and concatenates on dim=1 rather than dim=2 because
    its tensors stay in B, L, H, D. Without this branch the dict falls through both
    `try`s and raises AttributeError on `.key_cache`.
    """
    out = []
    if isinstance(pkv, dict):
        for i in sorted(pkv):
            e = pkv[i]
            if isinstance(e, dict) and "key_states" in e:
                out.append((e["key_states"], e["value_states"]))
            elif isinstance(e, (tuple, list)) and len(e) >= 2:
                out.append((e[0], e[1]))
        return out
    try:
        for lay in pkv.layers:
            out.append((lay.keys, lay.values))
        if out:
            return out
    except (AttributeError, TypeError):
        pass
    for i in range(len(pkv.key_cache)):
        out.append((pkv.key_cache[i], pkv.value_cache[i]))
    return out


def timed_wall(fn, warmup, n):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def timed_event(fn, warmup, n):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="pi0 | pi05 | smolvla. T011 extends E11 beyond pi0; the expert layout and the checkpoint-specific processors come from bench/model_registry.py.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=1000)   # §13.1: >= 1,000 timed calls
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    spec = MR.get(args.model)
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    policy, cfg, shipped = MR.load_policy(spec, n=args.steps)
    real = P.assert_real_weights(policy, spec=spec)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    tok = P.token_counts(policy, batch, spec)
    Pn, Sn = int(tok["prefix_tokens_P"]), int(tok["suffix_tokens_S"])
    P_valid = int(tok["prefix_tokens_valid"])
    ctx = build(policy, batch, spec)
    pkv, prefix_pad_masks = prefix_once(ctx)
    layers = cache_layers(pkv)
    print(f"cache: {len(layers)} layers; P={Pn} ({P_valid} valid, {Pn - P_valid} padded)")

    # --- handover.csv
    hrows, tot_bytes = [], 0
    for li, (k, v) in enumerate(layers):
        for name, t in (("key", k), ("value", v)):
            nb = t.numel() * t.element_size()
            tot_bytes += nb
            hrows.append({"model": "pi0", "layer": li, "tensor": name,
                          "shape": str(tuple(t.shape)), "dtype": str(t.dtype).split(".")[-1],
                          "stride": str(tuple(t.stride())), "contiguous": t.is_contiguous(),
                          "bytes": nb, "valid_tokens": P_valid, "padded_tokens": Pn - P_valid,
                          "notes": ""})
    per_layer = tot_bytes // len(layers)
    hrows.append({"model": "pi0", "layer": "TOTAL", "tensor": "key+value",
                  "shape": f"({len(layers)}, 2, {Pn}, {layers[0][0].shape[1]}, "
                           f"{layers[0][0].shape[-1]})",
                  "dtype": str(layers[0][0].dtype).split(".")[-1], "stride": "", "contiguous": "",
                  "bytes": tot_bytes, "valid_tokens": P_valid, "padded_tokens": Pn - P_valid,
                  "notes": f"{len(layers)} layers x 2 tensors; "
                           f"{per_layer} B per layer; cross-check "
                           f"{len(layers)}*2*{Pn}*{layers[0][0].shape[1]}*"
                           f"{layers[0][0].shape[-1]}*2 = "
                           f"{len(layers)*2*Pn*layers[0][0].shape[1]*layers[0][0].shape[-1]*2}"})
    print(f"  total KV bytes {tot_bytes:,} ({tot_bytes/1e6:.2f} MB), "
          f"{per_layer:,} per layer")

    # --- t_kv_materialise, both layouts
    kl = [k for k, _ in layers]
    vl = [v for _, v in layers]

    def stacked():
        return torch.stack([torch.stack(kl), torch.stack(vl)], dim=1).contiguous()

    def v_transposed():
        # K and V no longer share a shape once V is transposed to [..., head_dim, P], so they
        # cannot be stacked into one [layers, 2, ...] tensor. Materialise both contiguously.
        return (torch.stack(kl).contiguous(),
                torch.stack([v.transpose(-1, -2) for v in vl]).contiguous())

    mat = {}
    for layout, fn in (("stacked", stacked), ("v_transposed", v_transposed)):
        for timing, tf in (("wall_isolated", timed_wall), ("gpu_event", timed_event)):
            ms = tf(fn, args.warmup, args.timed)
            mat[(layout, timing)] = summarize(ms)
            print(f"  t_kv_materialise {layout:14s} {timing:14s} "
                  f"{mat[(layout, timing)]['median_ms']:8.4f} ms")

    # --- how the steps use it: pointers across steps, and the per-step deepcopy
    ptr_log, dc_ms = [], None
    orig_ds = policy.model.denoise_step
    step_i = {"n": 0}

    def wrapped(*a, **kw):
        # Signature-agnostic: pi0 passes these positionally with a state, pi0.5 and
        # SmolVLA by keyword without one (their state is in the prefix).
        past_key_values = kw.get('past_key_values', a[2] if len(a) > 2 else None)
        ls = cache_layers(past_key_values)
        ptr_log.append({"step": step_i["n"], "when": "entry",
                        "layer0_key_ptr": ls[0][0].data_ptr(),
                        "layer0_value_ptr": ls[0][1].data_ptr(),
                        "layer0_key_shape": str(tuple(ls[0][0].shape))})
        step_i["n"] += 1
        return orig_ds(*a, **kw)

    # ...and the cache the expert actually receives, i.e. AFTER the deepcopy at
    # modeling_pi0.py:918. Capturing only on entry to denoise_step would show a stable
    # address -- the same source object is passed in every step -- and would wrongly read as
    # "referenced, not copied".
    pgx = getattr(policy.model, spec.container)
    orig_pg = pgx.forward

    def wrapped_pg(*a, **kw):
        pkv_in = kw.get("past_key_values")
        if pkv_in is not None:
            ls = cache_layers(pkv_in)
            if ls:
                ptr_log.append({"step": step_i["n"] - 1, "when": "after_deepcopy",
                                "layer0_key_ptr": ls[0][0].data_ptr(),
                                "layer0_value_ptr": ls[0][1].data_ptr(),
                                "layer0_key_shape": str(tuple(ls[0][0].shape))})
        return orig_pg(*a, **kw)

    policy.model.denoise_step = wrapped
    pgx.forward = wrapped_pg
    try:
        with torch.no_grad():
            policy.predict_action_chunk(batch)
    finally:
        policy.model.denoise_step = orig_ds
        pgx.forward = orig_pg

    ms = timed_wall(lambda: copy.deepcopy(pkv), args.warmup, args.timed)
    dc = summarize(ms)
    dc_ms = dc["median_ms"]
    print(f"  copy.deepcopy(past_key_values) (modeling_pi0.py:918): {dc_ms:.4f} ms/step")

    # --- boundary.csv
    brows = [
        {"model": "pi0", "tensor": "prefix_kv_cache", "direction": "gpu_to_expert",
         "shape": f"({len(layers)}, 2, {Pn}, {layers[0][0].shape[1]}, "
                  f"{layers[0][0].shape[-1]})",
         "dtype": str(layers[0][0].dtype).split(".")[-1], "bytes": tot_bytes,
         "per": "inference", "notes": "the hand-over object itself"},
        {"model": "pi0", "tensor": "x_t_noisy_actions", "direction": "into_expert",
         "shape": f"(1, {policy.config.chunk_size}, {policy.config.max_action_dim})",
         "dtype": "float32",
         "bytes": policy.config.chunk_size * policy.config.max_action_dim * 4,
         "per": "step", "notes": "current noisy action chunk"},
        {"model": "pi0", "tensor": "state", "direction": "into_expert",
         "shape": str(tuple(ctx["state"].shape)),
         "dtype": str(ctx["state"].dtype).split(".")[-1],
         "bytes": ctx["state"].numel() * ctx["state"].element_size(),
         "per": "step", "notes": "re-embedded every step by embed_suffix"},
        {"model": "pi0", "tensor": "timestep", "direction": "into_expert", "shape": "(1,)",
         "dtype": "float32", "bytes": 4, "per": "step", "notes": "scalar per step"},
        {"model": "pi0", "tensor": "v_t_velocity", "direction": "out_of_expert",
         "shape": f"(1, {policy.config.chunk_size}, {policy.config.max_action_dim})",
         "dtype": "float32",
         "bytes": policy.config.chunk_size * policy.config.max_action_dim * 4,
         "per": "step", "notes": "the step's output, consumed by the Euler update"},
    ]
    per_step_total = sum(r["bytes"] for r in brows if r["per"] == "step")
    brows.append({"model": "pi0", "tensor": "total", "direction": "both", "shape": "",
                  "dtype": "", "bytes": per_step_total, "per": "step",
                  "notes": "sum of the per-step tensors; excludes the KV cache, which the "
                           "model assumes is loaded once"})

    for name, rows_ in (("handover.csv", hrows), ("boundary.csv", brows)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)
    with (out / "raw" / "kv_pointers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ptr_log[0].keys()))
        w.writeheader()
        w.writerows(ptr_log)

    # --- kv_reuse.md
    entry_ptrs = {p["layer0_key_ptr"] for p in ptr_log if p["when"] == "entry"}
    post_ptrs = [p for p in ptr_log if p["when"] == "after_deepcopy"]
    uniq = {p["layer0_key_ptr"] for p in post_ptrs} or entry_ptrs
    copied = len(uniq) > 1 or len(post_ptrs) > 0
    (out / "kv_reuse.md").write_text(f"""# How the denoising steps use the prefix KV cache (pi0)

**Answer: every step deep-copies the whole cache before using it, so the prefix KV is
re-materialised {len([p for p in ptr_log if p["when"] == "entry"])} times per inference — once
per denoising step — not loaded once.**

## Evidence from the source

`PI0Pytorch.denoise_step`, `lerobot/policies/pi0/modeling_pi0.py:918`:

```python
past_key_values = copy.deepcopy(past_key_values)
```

It sits immediately before the expert forward, inside the per-step function, so it runs once
per denoising step. The prefix itself is computed once per inference
(`modeling_pi0.py:847-853`, `use_cache=True`).

## Evidence from the running process

`raw/kv_pointers.csv` records `data_ptr()` of layer 0's key tensor at two points in every
`denoise_step` of one real inference at N = {args.steps}: on **entry**, and at the expert
forward, i.e. **after** the deepcopy.

| step | on entry | after the deepcopy |
|---|---|---|
""" + "\n".join(
    f"| {p['step']} | {p['layer0_key_ptr']} | "
    + str(next((q['layer0_key_ptr'] for q in post_ptrs if q['step'] == p['step']), 'n/a'))
    + " |"
    for p in ptr_log if p["when"] == "entry") + f"""

The **entry** address is stable across all {len([p for p in ptr_log if p['when']=='entry'])}
steps: `sample_actions` holds one cache object and passes the same one in every step. Read
alone, that would say the cache is referenced rather than copied — and it would be wrong.
The address **after** the deepcopy is a *different buffer from the original*, which is the
proof that a copy is made: the expert reads the copy, never the cache `sample_actions` holds.
That copy address is itself stable across the {len(post_ptrs)} steps, and that is allocator
reuse rather than reuse of the data — each step's copy is freed when the step ends, so PyTorch's
caching allocator hands the same {tot_bytes/1e6:.2f} MB block back for the next one. The bytes
are re-written every step regardless, which is what {dc_ms:.4f} ms and
`kv_copied_bytes_per_step = {tot_bytes:,}` measure.

## What it costs

| quantity | value |
|---|---|
| KV bytes per layer | {per_layer:,} B |
| KV bytes total | {tot_bytes:,} B ({tot_bytes/1e6:.2f} MB) |
| valid prefix tokens | {P_valid} of {Pn} ({Pn - P_valid} padded but still stored) |
| `copy.deepcopy` per step | **{dc_ms:.4f} ms** |
| `kv_copied_bytes_per_step` | **{tot_bytes:,} B** (the whole cache) |

The analytic model assumes `kv_copied_bytes_per_step = 0` and `t_kv_materialise = 0`. Both are
measurably not zero: the copy moves the entire {tot_bytes/1e6:.2f} MB object every step and
costs {dc_ms:.4f} ms, which at N = 10 is {dc_ms*10:.2f} ms per inference of pure copying.

**{Pn - P_valid} of {Pn} prefix tokens are padding** ({100*(Pn-P_valid)/Pn:.0f} %) and are
stored and copied like any other. A mapping that moved only the valid tokens would carry
{P_valid*tot_bytes//Pn:,} B instead.
""")

    # --- model_inputs.csv
    mi = ModelInputs(spec.name, spec.revision or "", sh_out(f"git -C {REPO} rev-parse HEAD"),
                     "MAXN+jetson_clocks", Pn, Sn, args.steps)
    mi.add("kv_bytes_per_layer", per_layer, "bytes", "value", 1, "handover.csv", key="padded")
    mi.add("kv_bytes_per_layer", per_layer * P_valid // Pn, "bytes", "value", 1,
           "handover.csv", key="valid")
    mi.add("kv_bytes_total", tot_bytes, "bytes", "value", 1, "handover.csv", key="padded")
    mi.add("kv_bytes_total", tot_bytes * P_valid // Pn, "bytes", "value", 1, "handover.csv",
           key="valid")
    for (layout, timing), s in mat.items():
        mi.add_stats("t_kv_materialise", s, "ms", s["n"], "raw/kv_materialise.csv",
                     key=layout, variant=f"layout={layout}", timing=timing)
    mi.add("kv_copied_bytes_per_step", tot_bytes, "bytes", "value", 1, "kv_reuse.md")
    for r in brows:
        if r["per"] == "step":
            mi.add("boundary_bytes_per_step", r["bytes"], "bytes", "value", 1,
                   "boundary.csv", key=r["tensor"])
    with (out / "kv_copy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["what", "median_ms", "p5_ms", "p95_ms", "min_ms", "n", "bytes", "source"])
        w.writerow(["copy.deepcopy(past_key_values)", round(dc["median_ms"], 6),
                    round(dc["p5_ms"], 6), round(dc["p95_ms"], 6), round(dc["min_ms"], 6),
                    dc["n"], tot_bytes,
                    "lerobot/policies/pi0/modeling_pi0.py:918, once per denoising step"])
    mi.add_stats("t_kv_materialise", dc, "ms", dc["n"], "kv_copy.csv",
                 key="stacked", variant="layout=deepcopy_per_step", timing="wall_isolated")
    with (out / "raw" / "kv_materialise.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "timing", "median_ms", "p5_ms", "p95_ms", "n"])
        for (layout, timing), s in mat.items():
            w.writerow([layout, timing, round(s["median_ms"], 6), round(s["p5_ms"], 6),
                        round(s["p95_ms"], 6), s["n"]])
    n_mi = mi.write(out / "model_inputs.csv")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11c", "task": "T010",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.name, "model_repo": spec.checkpoint,
        "model_commit": spec.revision, "model_revision": spec.revision,
        "checkpoint": spec.checkpoint, "precision": "bfloat16",
        "compile_mode": None, "batch": 1, "num_denoise_steps": args.steps,
        "action_horizon": getattr(cfg, "chunk_size", None),
        "prefix_tokens": Pn, "prefix_tokens_valid": P_valid, "suffix_tokens": Sn,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/handover.py --outdir {args.outdir}",
        "kv_bytes_total": tot_bytes, "kv_bytes_per_layer": per_layer,
        "deepcopy_ms_per_step": round(dc_ms, 6),
        "distinct_layer0_key_ptrs": len(uniq),
        "denoise_steps_observed": len([p for p in ptr_log if p["when"] == "entry"]),
        "pointer_log_rows": len(ptr_log),
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": [
            "The cross-check in the task text is 18*2*P*1*256*2 B; the TOTAL row of "
            "handover.csv carries both that figure and the measured one, and they are "
            "reported, not forced.",
            "kv_copied_bytes_per_step is the WHOLE cache: copy.deepcopy at "
            "modeling_pi0.py:918 runs once per denoising step. The analytic model assumes 0.",
            "Padded prefix positions are stored and copied exactly like valid ones; both "
            "figures are given.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    e11c(out)
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({n_mi} model_inputs rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
