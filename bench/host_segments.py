"""T010 / E11a — the action expert's *host* segments, timed on real inputs.

Protocol §13.2 defines a host segment as a maximal run of consecutive operations of one
action-expert layer that is **neither a large linear** (q/k/v/o, MLP gate/up/down) **nor an
attention matrix product**. Those are the parts a PIM design leaves on the GPU, and today the
research side's model prices all of them with one analytic formula
(`docs/00_context.md`): `max(FLOPs / 16.8 TFLOP/s, bytes / 143 GB/s) + kernels x 20 us`,
giving ~0.16 ms per layer. This measures them.

**Where the segments actually live.** During denoising, `PI0Pytorch.denoise_step`
(`modeling_pi0.py:894`) calls `paligemma_with_expert.forward` with `inputs_embeds=[None,
suffix_embs]`, which takes the `elif inputs_embeds[0] is None` branch
(`modeling_pi0.py:478`) and runs `gemma_expert.model.forward` — the ordinary decoder stack.
So the layer body is `pi_gemma.py:157-188`, NOT `compute_layer_complete` (which serves the
joint prefix+suffix path only). Every segment below cites that path.

Two things the loaded model does that `docs/01_measurement_plan.md`'s segment table does not
say, and the model wins (protocol: "the model wins"; recorded in `segments.csv.notes`):

  * `embed_suffix` returns **`adarms_cond = None`** for pi0 (`modeling_pi0.py:734`), so
    `PiGemmaRMSNorm.forward` takes its unconditioned branch — `normed * (1.0 + weight.float())`
    (`pi_gemma.py:111-135`). There is no AdaRMS `dense(cond)`/`chunk`/scale-shift-gate work in
    pi0, and the residuals are `x + y`, not `x + y*gate`. pi05 is expected to differ.
  * `eager_attention_forward` calls `repeat_kv` (`modeling_gemma.py:205-206`), materialising
    the single KV head into 8 before the matmuls. That expansion is real traffic the analytic
    model does not price.

Inputs are the model's own tensors: one real denoising step of the fixture observation is run
with the layer wrapped, the layer's arguments are captured at layer 0 and layer 9, and every
intermediate is then reproduced by running the model's own modules in order. Segments are timed
by calling the model's own callables wherever one exists; only what the model does *inline* is
re-implemented, line for line, with the file and line cited.

    python bench/host_segments.py --outdir results/<date>_E11a_pi0_hostseg
    python bench/host_segments.py --outdir /tmp/x --timed 20 --skip-sequence   # smoke test
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

LAYERS = (0, 9)


# --------------------------------------------------------------------------- capture


class LayerCapture:
    """Record one expert layer's own arguments during a real denoising step."""

    def __init__(self, policy, layer_idx: int):
        self.layer = policy.model.paligemma_with_expert.gemma_expert.model.layers[layer_idx]
        self.idx = layer_idx
        self.args = None
        self._orig = None

    def __enter__(self):
        self._orig = self.layer.forward
        cap = self

        def wrapper(hidden_states, attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False, cache_position=None,
                    position_embeddings=None, adarms_cond=None, **kw):
            if cap.args is None:                       # first denoising step only
                # The prefix K/V for THIS layer, read before the attention updates the cache,
                # so it holds the P prefix tokens only. Without it the replayed attention
                # would run over S=51 keys instead of the real T = P + S = 867, which is a
                # 17x understatement -- the first version of this script did exactly that.
                pk = pv = None
                if past_key_values is not None:
                    li_ = cap.idx
                    try:
                        lay = past_key_values.layers[li_]
                        pk, pv = lay.keys, lay.values
                    except (AttributeError, IndexError, TypeError):
                        try:
                            pk = past_key_values.key_cache[li_]
                            pv = past_key_values.value_cache[li_]
                        except Exception:                          # noqa: BLE001
                            pk = pv = None
                cap.args = {
                    "hidden_states": hidden_states.detach().clone(),
                    "attention_mask": None if attention_mask is None
                    else attention_mask.detach().clone(),
                    "position_embeddings": None if position_embeddings is None else
                    tuple(t.detach().clone() for t in position_embeddings),
                    "adarms_cond": None if adarms_cond is None else adarms_cond.detach().clone(),
                    "prefix_k": None if pk is None else pk.detach().clone(),
                    "prefix_v": None if pv is None else pv.detach().clone(),
                }
            return cap._orig(hidden_states, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             use_cache=use_cache, cache_position=cache_position,
                             position_embeddings=position_embeddings,
                             adarms_cond=adarms_cond, **kw)

        self.layer.forward = wrapper
        return self

    def __exit__(self, *exc):
        self.layer.forward = self._orig
        return False


def intermediates(policy, cap: LayerCapture) -> dict:
    """Reproduce every tensor the layer body passes between segments, using the model's own
    modules, in the order of `pi_gemma.py:169-188`."""
    from transformers.models.gemma import modeling_gemma as MG

    layer = cap.layer
    a = cap.args
    x = a["hidden_states"]
    attn = layer.self_attn
    with torch.no_grad():
        normed1, gate1 = layer.input_layernorm(x, cond=a["adarms_cond"])     # pi_gemma.py:171
        shape = (*normed1.shape[:-1], -1, attn.head_dim)                     # gemma:256-257
        q = attn.q_proj(normed1).view(shape).transpose(1, 2)
        k = attn.k_proj(normed1).view(shape).transpose(1, 2)
        v = attn.v_proj(normed1).view(shape).transpose(1, 2)
        cos, sin = a["position_embeddings"]
        q_r, k_r = MG.apply_rotary_pos_emb(q, k, cos, sin)                   # gemma:265
        # The layer attends over the FULL keys/values: the P cached prefix tokens followed by
        # this step's S suffix tokens, which is what `past_key_values.update()` builds
        # (modeling_gemma.py:268-270). Rebuild it the same way, so T = P + S.
        kv_groups = attn.num_key_value_groups
        if a["prefix_k"] is not None:
            k_full = torch.cat([a["prefix_k"], k_r], dim=2)
            v_full = torch.cat([a["prefix_v"], v], dim=2)
        else:
            k_full, v_full = k_r, v
        scores = torch.matmul(q_r, MG.repeat_kv(k_full, kv_groups).transpose(2, 3)) * attn.scaling
        mask = a["attention_mask"]
        if mask is not None:
            mask = mask[..., : scores.shape[-1]]
        scores_masked = scores if mask is None else scores + mask           # gemma:211
        probs = torch.nn.functional.softmax(scores_masked, dim=-1, dtype=torch.float32)
        probs_cast = probs.to(q_r.dtype)                                     # gemma:213
        ctx = torch.matmul(probs_cast, MG.repeat_kv(v_full, kv_groups)).transpose(1, 2).contiguous()
        attn_out = attn.o_proj(ctx.reshape(*x.shape[:-1], -1).contiguous())  # gemma:288-289
        resid1 = attn_out + x                                                # pi_gemma:182 (gate None)
        normed2, gate2 = layer.post_attention_layernorm(resid1, cond=a["adarms_cond"])
        g = layer.mlp.gate_proj(normed2)
        u = layer.mlp.up_proj(normed2)
        mlp_out = layer.mlp.down_proj(layer.mlp.act_fn(g) * u)               # gemma:81
    return {"x": x, "normed1": normed1, "gate1": gate1, "q": q, "k": k, "v": v,
            "cos": cos, "sin": sin, "q_r": q_r, "k_r": k_r,
            "scores": scores_masked, "scores_fp32": scores_masked.float(),
            "k_full": k_full, "v_full": v_full,
            "probs_cast": probs_cast, "ctx": ctx, "attn_out": attn_out,
            "resid1": resid1, "normed2": normed2, "gate2": gate2,
            "g": g, "u": u, "mlp_out": mlp_out, "mask": mask,
            "attn": attn, "layer": layer, "adarms_cond": a["adarms_cond"],
            "kv_groups": kv_groups}


# --------------------------------------------------------------------------- segments


def build_segments(t: dict, S: int, D: int, I: int, H: int, HD: int, T_keys: int) -> list[dict]:
    """The host segments of one expert layer, in order, each with its own callable.

    `source_ref` is `file:line` in the installed packages at the pinned commits.
    `flops_formula` and the byte counts are analytic, from these real shapes.
    """
    from transformers.models.gemma import modeling_gemma as MG
    import lerobot.policies.pi_gemma as PG

    layer, attn, cond = t["layer"], t["attn"], t["adarms_cond"]
    B2 = 2                                                    # bf16
    segs = [
        dict(segment="rmsnorm1", order=0, in_model="yes",
             source_ref="lerobot/policies/pi_gemma.py:111-135 (call site :171)",
             ops="PiGemmaRMSNorm: x.float(); mean(x^2); rsqrt; mul; * (1 + weight.float()); to(bf16)",
             fn=lambda: layer.input_layernorm(t["x"], cond=cond),
             inputs=f"x:{tuple(t['x'].shape)}:{t['x'].dtype}",
             outputs=f"normed:{tuple(t['normed1'].shape)}:{t['normed1'].dtype}",
             bytes_read=S * D * B2 + D * B2, bytes_written=S * D * B2,
             flops=8 * S * D, flops_formula="8*S*D (square, mean, rsqrt, mul, weighted mul)",
             notes="adarms_cond is None for pi0 (modeling_pi0.py:734) so the unconditioned "
                   "branch runs; pi05 is expected to take the AdaRMS branch instead"),
        dict(segment="rope", order=1, in_model="yes",
             source_ref="transformers/models/gemma/modeling_gemma.py:158-180 (call site :265)",
             ops="apply_rotary_pos_emb on q and k: unsqueeze; q*cos; rotate_half(q) (slice+neg+cat); *sin; add",
             fn=lambda: MG.apply_rotary_pos_emb(t["q"], t["k"], t["cos"], t["sin"]),
             inputs=f"q:{tuple(t['q'].shape)}:{t['q'].dtype};k:{tuple(t['k'].shape)}:{t['k'].dtype}",
             outputs=f"q_r:{tuple(t['q_r'].shape)};k_r:{tuple(t['k_r'].shape)}",
             bytes_read=S * (H + 1) * HD * B2 * 2, bytes_written=S * (H + 1) * HD * B2,
             flops=6 * S * (H * HD + HD),
             flops_formula="6*S*(H*head_dim + kv_heads*head_dim); q and k both rotated",
             notes="applied to q AND k; rotate_half does a torch.cat, so >= 3 kernels each"),
        dict(segment="softmax", order=2, in_model="yes",
             source_ref="transformers/models/gemma/modeling_gemma.py:211-213",
             ops="scores + attention_mask; softmax(dim=-1, dtype=float32); .to(bf16)",
             fn=lambda: torch.nn.functional.softmax(
                 t["scores"], dim=-1, dtype=torch.float32).to(t["q_r"].dtype),
             inputs=f"scores:{tuple(t['scores'].shape)}:{t['scores'].dtype}",
             outputs=f"probs:{tuple(t['probs_cast'].shape)}:{t['probs_cast'].dtype}",
             bytes_read=H * S * T_keys * B2, bytes_written=H * S * T_keys * B2,
             flops=5 * H * S * T_keys, flops_formula="5*H*S*T (max, sub, exp, sum, div)",
             notes="the model's softmax is fp32-internal with a cast back to bf16, not fp16; "
                   "the mask add is fused into this segment per the plan's definition"),
        dict(segment="residual_rmsnorm2", order=3, in_model="yes",
             source_ref="lerobot/policies/pi_gemma.py:182-186 (_gated_residual at :54-67)",
             ops="x + attn_out (gate None); then PiGemmaRMSNorm",
             fn=lambda: layer.post_attention_layernorm(
                 PG._gated_residual(t["x"], t["attn_out"], t["gate1"]), cond=cond),
             inputs=f"x:{tuple(t['x'].shape)};attn_out:{tuple(t['attn_out'].shape)}",
             outputs=f"normed2:{tuple(t['normed2'].shape)}",
             bytes_read=3 * S * D * B2, bytes_written=2 * S * D * B2,
             flops=9 * S * D, flops_formula="S*D (add) + 8*S*D (rmsnorm)",
             notes="gate is None for pi0, so _gated_residual reduces to x + y"),
        dict(segment="gelu_gate_mul_up", order=4, in_model="yes",
             source_ref="transformers/models/gemma/modeling_gemma.py:81",
             ops="act_fn(gate_proj_out) * up_proj_out; act_fn = gelu_pytorch_tanh",
             fn=lambda: layer.mlp.act_fn(t["g"]) * t["u"],
             inputs=f"gate:{tuple(t['g'].shape)}:{t['g'].dtype};up:{tuple(t['u'].shape)}",
             outputs=f"h:{tuple(t['g'].shape)}",
             bytes_read=2 * S * I * B2, bytes_written=S * I * B2,
             flops=9 * S * I, flops_formula="9*S*I (tanh-approx GeLU ~8 flops + 1 multiply)",
             notes="the three projections around it are large linears and are NOT part of the "
                   "segment (protocol §13.2)"),
        dict(segment="partials_residual", order=5, in_model="mapping_only",
             source_ref="docs/01_measurement_plan.md segment table (no such tensor in the model)",
             ops="partials.sum(0) + x, partials [4, S, D]",
             fn=lambda: t["_partials"].sum(0) + t["x"][0],
             inputs=f"partials:(4, {S}, {D}):bf16;x:{tuple(t['x'].shape)}",
             outputs=f"out:({S}, {D})",
             bytes_read=5 * S * D * B2, bytes_written=S * D * B2,
             flops=4 * S * D, flops_formula="4*S*D (3 adds to reduce 4 partials + 1 residual add)",
             notes="mapping_only: the PIM design splits the down projection into 4 partial "
                   "results; the model has no such tensor, so random bf16 values are used"),
        dict(segment="residual", order=6, in_model="yes",
             source_ref="lerobot/policies/pi_gemma.py:54-67 (call sites :182, :188)",
             ops="x + y (gate is None for pi0)",
             fn=lambda: PG._gated_residual(t["x"], t["attn_out"], None),
             inputs=f"x:{tuple(t['x'].shape)};y:{tuple(t['attn_out'].shape)}",
             outputs=f"out:{tuple(t['x'].shape)}",
             bytes_read=2 * S * D * B2, bytes_written=S * D * B2,
             flops=S * D, flops_formula="S*D",
             notes="the plain-residual variant, without the partial sums"),
        dict(segment="attention_gpu", order=7, in_model="yes",
             source_ref="transformers/models/gemma/modeling_gemma.py:195-218",
             ops="repeat_kv(k,v) x8; scores = q@K^T * scaling; +mask; softmax fp32; cast; probs@V; transpose",
             fn=lambda: MG.eager_attention_forward(
                 attn, t["q_r"], t["k_full"], t["v_full"], t["mask"], attn.scaling),
             inputs=f"q:{tuple(t['q_r'].shape)};k:{tuple(t['k_full'].shape)};"
                    f"v:{tuple(t['v_full'].shape)}",
             outputs=f"ctx:{tuple(t['ctx'].shape)}",
             bytes_read=S * H * HD * B2 + 2 * T_keys * HD * B2,
             bytes_written=S * H * HD * B2,
             flops=2 * 2 * H * S * T_keys * HD,
             flops_formula="2*(2*H*S*T*head_dim): QK^T and probs@V",
             notes="design variant: the whole attention on the GPU. repeat_kv materialises the "
                   "single KV head into H=8 before both matmuls; unique KV bytes counted once"),
        dict(segment="cast_scores_fp32_fp16", order=8, in_model="mapping_only",
             source_ref="docs/01_measurement_plan.md segment table; the model's own cast is the "
                        "bf16 one inside softmax at modeling_gemma.py:213",
             ops="scores_fp32.to(float16)",
             fn=lambda: t["scores_fp32"].to(torch.float16),
             inputs=f"scores:{tuple(t['scores_fp32'].shape)}:float32",
             outputs=f"scores:{tuple(t['scores_fp32'].shape)}:float16",
             bytes_read=H * S * T_keys * 4, bytes_written=H * S * T_keys * 2,
             flops=0, flops_formula="0 (pure dtype conversion)",
             notes="the boundary traffic of the mapping: fp32 scores out, fp16 back. The model "
                   "itself casts fp32->bf16 inside softmax, so this is mapping_only"),
    ]
    return segs


# --------------------------------------------------------------------------- timing


def time_segment(fn, timing: str, warmup: int, n: int, gap_ms: float = 0.0) -> list[float]:
    """Protocol §13.1's three timing modes."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    if timing == "gpu_event":
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
        for a, b in ev:
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        return [a.elapsed_time(b) for a, b in ev]
    for _ in range(n):
        if gap_ms:
            time.sleep(gap_ms / 1e3)
        if timing == "wall_isolated":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        elif timing == "wall_launch":
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1e3)
        else:
            raise ValueError(timing)
    if timing == "wall_launch":
        torch.cuda.synchronize()
    return out


def make_style(fn, style: str):
    """Return (callable, note) for an exec_style, or (None, error) if it cannot be built."""
    if style == "eager":
        return fn, ""
    if style == "compiled":
        try:
            c = torch.compile(fn, dynamic=False)
            for _ in range(3):
                c()
            torch.cuda.synchronize()
            return c, ""
        except Exception as e:                                    # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:160]}"
    if style == "cudagraph":
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            return (lambda g=g: g.replay()), ""
        except Exception as e:                                    # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:160]}"
    raise ValueError(style)


def count_kernels(fn, n: int = 10) -> int:
    """Kernels per call, from one torch.profiler run over n calls (§13.1)."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        k = sum(1 for e in pr.events() for _ in (getattr(e, "kernels", []) or []))
        if k == 0:                       # CUPTI needs root here; fall back to the CUDA total
            k = sum(int(e.count) for e in pr.key_averages() if e.device_time_total > 0)
        return round(k / n, 2)
    except Exception:                                             # noqa: BLE001
        return -1


# --------------------------------------------------------------------------- model_inputs.csv

MI_HEADER = ["model", "model_commit", "bench_commit", "quantity", "key", "variant", "value",
             "unit", "stat", "n", "exec_style", "timing", "power_mode", "precision",
             "P", "S", "N", "source_file"]


class ModelInputs:
    """Emitter for protocol §13.8. The checker is strict and is run in place, never adapted:
    the header must match exactly, `quantity` comes from a closed table, `precision` must be
    `bf16` (our meta.json says `bfloat16`, which would fail), `variant` must match
    `^[A-Za-z0-9_]+=[^;=]+(;...)*$`, `value` must be finite — an unmeasured quantity is left
    OUT of the file and reported in summary.md — and `source_file` must exist in the result
    directory.
    """

    def __init__(self, model, model_commit, bench_commit, power_mode, P, S, N):
        self.rows = []
        self.base = dict(model=model, model_commit=model_commit, bench_commit=bench_commit,
                         power_mode=power_mode, precision="bf16", P=P, S=S, N=N)

    def add(self, quantity, value, unit, stat, n, source_file, *, key="", variant="",
            exec_style="", timing=""):
        if value is None or value == "":
            return                                   # §13.8: leave unmeasured rows out
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            return                                   # §13.8: leave unmeasured rows out
        r = dict(self.base)
        r.update(quantity=quantity, key=key, variant=variant, value=repr(round(v, 9)),
                 unit=unit, stat=stat, n=int(n), exec_style=exec_style, timing=timing,
                 source_file=source_file)
        r["value"] = f"{v:.9g}"
        self.rows.append([r[c] for c in MI_HEADER])

    def add_stats(self, quantity, s, unit, n, source_file, **kw):
        for stat, key_name in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
            self.add(quantity, s[key_name], unit, stat, n, source_file, **kw)

    def write(self, path: Path):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(MI_HEADER)
            w.writerows(self.rows)
        return len(self.rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--styles", default="eager,compiled,cudagraph")
    ap.add_argument("--mode", choices=("time", "count"), default="time",
                    help="`count` only counts kernels per call with torch.profiler and writes "
                         "kernels_per_call.csv; it needs root for CUPTI on this device. "
                         "`time` does the timing and never attaches a profiler, because CUPTI "
                         "inflates every subsequent call by ~30 %. Run `count` first, then "
                         "`time` with --kernels-csv pointing at its output.")
    ap.add_argument("--kernels-csv", default=None,
                    help="kernels_per_call.csv from a previous --mode count run")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    styles = [s for s in args.styles.split(",") if s]

    kernels_map = {}
    if args.kernels_csv and Path(args.kernels_csv).exists():
        for r in csv.DictReader(Path(args.kernels_csv).open()):
            kernels_map[(r["segment"], r["exec_style"])] = float(r["n_kernels"])
        print(f"kernel counts: {len(kernels_map)} entries from {args.kernels_csv}")

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    policy, cfg, shipped = P.load_policy(num_inference_steps=args.steps)
    real = P.assert_real_weights(policy)
    print("weights:", json.dumps(real), flush=True)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    tok = P.token_counts(policy, batch)
    Pn, Sn = int(tok["prefix_tokens_P"]), int(tok["suffix_tokens_S"])
    print("tokens:", json.dumps(tok), flush=True)

    # --- capture the real layer arguments during one denoising step
    caps = {}
    for li in LAYERS:
        with LayerCapture(policy, li) as cap, torch.no_grad():
            policy.predict_action_chunk(batch)
        if cap.args is None:
            raise SystemExit(f"layer {li} was never called — wrap point wrong")
        caps[li] = cap
    torch.cuda.synchronize()

    ex = policy.model.paligemma_with_expert.gemma_expert
    D = ex.config.hidden_size
    I = ex.config.intermediate_size
    H = ex.config.num_attention_heads
    HD = ex.config.head_dim
    all_rows, seg_rows, layer_rows, samples = [], [], [], []
    mi = ModelInputs("pi0", P.REVISION, sh_out(f"git -C {REPO} rev-parse HEAD"),
                     "MAXN+jetson_clocks", Pn, Sn, args.steps)
    notes_not_measured = []

    for li in LAYERS:
        t = intermediates(policy, caps[li])
        t["_partials"] = torch.randn(4, Sn, D, device="cuda", dtype=t["x"].dtype)
        T_keys = t["scores"].shape[-1]
        print(f"\nlayer {li}: x{tuple(t['x'].shape)} scores{tuple(t['scores'].shape)} "
              f"T={T_keys} D={D} I={I} H={H} head_dim={HD}", flush=True)
        segs = build_segments(t, Sn, D, I, H, HD, T_keys)

        for sd in segs:
            if li == LAYERS[0]:
                seg_rows.append({
                    "model": "pi0", "segment": sd["segment"], "order_in_layer": sd["order"],
                    "layer_kind": "expert_self_attn", "source_ref": sd["source_ref"],
                    "ops": sd["ops"], "inputs": sd["inputs"], "outputs": sd["outputs"],
                    "bytes_read": sd["bytes_read"], "bytes_written": sd["bytes_written"],
                    "flops": sd["flops"], "flops_formula": sd["flops_formula"],
                    "in_model_code": sd["in_model"], "notes": sd["notes"]})
            for style in styles:
                fn, err = make_style(sd["fn"], style)
                if fn is None:
                    notes_not_measured.append(
                        f"{sd['segment']} layer={li} exec_style={style}: {err}")
                    print(f"  {sd['segment']:22s} {style:9s} NOT MEASURED: {err[:70]}")
                    continue
                if args.mode == "count":
                    nk = count_kernels(fn)
                    all_rows.append({"segment": sd["segment"], "layer": li,
                                     "exec_style": style, "n_kernels": nk})
                    print(f"  {sd['segment']:22s} {style:9s} kernels={nk}", flush=True)
                    continue
                nk = kernels_map.get((sd["segment"], style), 0.0)
                for timing, gap in (("wall_isolated", 0.0), ("wall_isolated", 2.0),
                                    ("gpu_event", 0.0), ("wall_launch", 0.0)):
                    ms = time_segment(fn, timing, args.warmup, args.timed, gap)
                    s = summarize(ms)
                    variant = f"layer={li}"
                    if timing == "wall_isolated":
                        variant += f";gap_ms={int(gap)}"
                    for i, v in enumerate(ms):
                        samples.append({"segment": sd["segment"], "exec_style": style,
                                        "timing": timing, "variant": variant,
                                        "iter": i, "ms": v})
                    row = {"segment": sd["segment"], "layer": li, "exec_style": style,
                           "timing": timing, "variant": variant,
                           "median_ms": round(s["median_ms"], 6),
                           "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                           "min_ms": round(s["min_ms"], 6), "n": s["n"], "n_kernels": nk,
                           "bytes": sd["bytes_read"] + sd["bytes_written"], "flops": sd["flops"],
                           "achieved_GBps": "", "achieved_TFLOPps": ""}
                    if timing == "gpu_event" and s["median_ms"] > 0:
                        sec = s["median_ms"] / 1e3
                        row["achieved_GBps"] = round(row["bytes"] / sec / 1e9, 3)
                        row["achieved_TFLOPps"] = round(sd["flops"] / sec / 1e12, 5)
                        mi.add("achieved_bw_GBps", row["achieved_GBps"], "GBps", "median",
                               s["n"], "segment_summary.csv", key=sd["segment"],
                               variant=variant, exec_style=style, timing="gpu_event")
                        if sd["flops"]:
                            mi.add("achieved_tflops", row["achieved_TFLOPps"], "TFLOPps",
                                   "median", s["n"], "segment_summary.csv", key=sd["segment"],
                                   variant=variant, exec_style=style, timing="gpu_event")
                    all_rows.append(row)
                    mi.add_stats("t_host_segment", s, "ms", s["n"], "segment_summary.csv",
                                 key=sd["segment"], variant=variant, exec_style=style,
                                 timing=timing)
                    if timing == "wall_launch" and nk > 0:
                        mi.add("t_launch_per_kernel", s["median_ms"] / nk, "ms", "median",
                               s["n"], "segment_summary.csv", key=sd["segment"],
                               variant=variant, exec_style=style, timing="wall_launch")
                if nk > 0:
                    mi.add("n_kernels_host_segment", nk, "count", "value", 1,
                           "segment_summary.csv", key=sd["segment"],
                           variant=f"layer={li};exec_style={style}", exec_style=style)
                print(f"  {sd['segment']:22s} {style:9s} "
                      f"{[r['median_ms'] for r in all_rows[-4:]]} ms  kernels={nk}", flush=True)
            if li == LAYERS[0]:
                mi.add("bytes_host_segment", sd["bytes_read"] + sd["bytes_written"], "bytes",
                       "value", 1, "segments.csv", key=sd["segment"], timing="analytic")
                mi.add("flops_host_segment", sd["flops"], "flops", "value", 1, "segments.csv",
                       key=sd["segment"], timing="analytic")

        # --- the per-layer sequence of the six mapping segments, and the whole-layer graph
        if not args.skip_sequence and args.mode == "time":
            six = [s for s in segs if s["segment"] in (
                "rmsnorm1", "rope", "softmax", "residual_rmsnorm2", "gelu_gate_mul_up",
                "partials_residual")]

            def seq():
                for s_ in six:
                    s_["fn"]()

            for style in ("eager", "compiled"):
                fn, err = make_style(seq, style)
                if fn is None:
                    notes_not_measured.append(f"sequence layer={li} {style}: {err}")
                    continue
                ms = time_segment(fn, "wall_isolated", args.warmup, max(200, args.timed // 5))
                s = summarize(ms)
                layer_rows.append({"exec_style": style, "key": "sequence",
                                   "variant": f"layer={li}", "median_ms": round(s["median_ms"], 6),
                                   "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                                   "n": s["n"]})
                mi.add_stats("t_host_layer", s, "ms", s["n"], "host_layer_summary.csv",
                             key="sequence", variant=f"layer={li}", exec_style=style,
                             timing="wall_isolated")
                print(f"  sequence {style:9s} {s['median_ms']:.4f} ms", flush=True)
            gfn, err = make_style(seq, "cudagraph")
            if gfn is None:
                notes_not_measured.append(f"whole_layer_graph layer={li}: {err}")
            else:
                ms = time_segment(gfn, "wall_isolated", args.warmup, max(200, args.timed // 5))
                s = summarize(ms)
                layer_rows.append({"exec_style": "cudagraph", "key": "whole_layer_graph",
                                   "variant": f"layer={li}", "median_ms": round(s["median_ms"], 6),
                                   "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                                   "n": s["n"]})
                mi.add_stats("t_host_layer", s, "ms", s["n"], "host_layer_summary.csv",
                             key="whole_layer_graph", variant=f"layer={li}",
                             exec_style="cudagraph", timing="wall_isolated")
                print(f"  whole_layer_graph      {s['median_ms']:.4f} ms", flush=True)
            # sum_of_segments is derived, per style
            for style in styles:
                tot = sum(r["median_ms"] for r in all_rows
                          if r["layer"] == li and r["exec_style"] == style
                          and r["timing"] == "wall_isolated" and "gap_ms=0" in r["variant"]
                          and r["segment"] in {s_["segment"] for s_ in six})
                if tot > 0:
                    layer_rows.append({"exec_style": style, "key": "sum_of_segments",
                                       "variant": f"layer={li}", "median_ms": round(tot, 6),
                                       "p5_ms": "", "p95_ms": "", "n": 6})
                    mi.add("t_host_layer", tot, "ms", "median", 6, "host_layer_summary.csv",
                           key="sum_of_segments", variant=f"layer={li}", exec_style=style,
                           timing="wall_isolated")
    if args.mode == "count":
        last = policy.model.paligemma_with_expert.gemma_expert.model.layers[-1]
        holder, orig_f = {}, last.forward

        def cap_last(*a, **k):
            r = orig_f(*a, **k)
            holder.setdefault("h", r[0] if isinstance(r, tuple) else r)
            return r

        last.forward = cap_last
        try:
            with torch.no_grad():
                policy.predict_action_chunk(batch)
        finally:
            last.forward = orig_f
        defs, err = build_extras(policy, batch, args, holder["h"].detach().clone())
        for name, fn, _ref, _ops in defs:
            for style in styles:
                f_, e = make_style(fn, style)
                if f_ is None:
                    continue
                nk = count_kernels(f_)
                all_rows.append({"segment": name, "layer": -1, "exec_style": style,
                                 "n_kernels": nk})
                print(f"  {name:22s} {style:9s} kernels={nk}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        f = out / "kernels_per_call.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["segment", "layer", "exec_style", "n_kernels"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {f} ({len(all_rows)} rows)")
        return 0
    return finish(out, args, gate, policy, cfg, shipped, real, tok, seg_rows, all_rows,
                  layer_rows, samples, mi, notes_not_measured, styles, Pn, Sn)


# --------------------------------------------------------------------------- per-step extras


class DenoiseCapture:
    """Record `denoise_step`'s own arguments, for the per-step extras."""

    def __init__(self, policy):
        self.model = policy.model
        self.args = None
        self._orig = None

    def __enter__(self):
        self._orig = self.model.denoise_step
        cap = self

        def wrapper(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if cap.args is None:
                cap.args = {"state": state.detach().clone(), "x_t": x_t.detach().clone(),
                            "timestep": timestep.detach().clone()}
            return cap._orig(state, prefix_pad_masks, past_key_values, x_t, timestep)

        self.model.denoise_step = wrapper
        return self

    def __exit__(self, *exc):
        self.model.denoise_step = self._orig
        return False


def build_extras(policy, batch, args, last_hidden):
    """The three per-step extras (`order_in_layer = -1`), as (name, callable, ref, ops)."""
    m = policy.model
    with DenoiseCapture(policy) as dc, torch.no_grad():
        policy.predict_action_chunk(batch)
    if dc.args is None:
        return [], "per-step extras: denoise_step was never called"
    st, x_t, ts = dc.args["state"], dc.args["x_t"], dc.args["timestep"]
    ex_norm = m.paligemma_with_expert.gemma_expert.model.norm
    chunk = m.config.chunk_size
    dt = -1.0 / args.steps
    v_t = torch.randn_like(x_t)

    def final_norm_out_proj():
        r = ex_norm(last_hidden)
        h = r[0] if isinstance(r, tuple) else r
        return m.action_out_proj(h[:, -chunk:].to(dtype=torch.float32))

    return [
        ("suffix_embed", lambda: m.embed_suffix(st, x_t, ts),
         "lerobot/policies/pi0/modeling_pi0.py:688-749",
         "state_proj; sinusoidal time embedding; action_in_proj; expand; cat; "
         "action_time_mlp_in; swish; action_time_mlp_out"),
        ("final_norm_out_proj", final_norm_out_proj,
         "lerobot/policies/pi0/modeling_pi0.py:929-932 (final norm inside pi_gemma.py:209)",
         "expert final RMSNorm; slice last chunk_size; cast to fp32; action_out_proj 1024->32"),
        ("euler_update", lambda: x_t + dt * v_t,
         "lerobot/policies/pi0/modeling_pi0.py:887", "x_t + dt * v_t"),
    ], None


def measure_extras(policy, batch, args, mi, all_rows, notes, styles, last_hidden, kernels):
    defs, err = build_extras(policy, batch, args, last_hidden)
    if err:
        notes.append(err)
        return []
    rows = []
    for name, fn, ref, ops in defs:
        for style in styles:
            f, e = make_style(fn, style)
            if f is None:
                notes.append(f"{name} exec_style={style}: {e}")
                print(f"  {name:22s} {style:9s} NOT MEASURED: {e[:70]}")
                continue
            nk = kernels.get((name, style), 0.0)
            for timing, gap in (("wall_isolated", 0.0), ("wall_isolated", 2.0),
                                ("gpu_event", 0.0), ("wall_launch", 0.0)):
                ms = time_segment(f, timing, args.warmup, args.timed, gap)
                s_ = summarize(ms)
                variant = "step_extra=1" + (f";gap_ms={int(gap)}"
                                            if timing == "wall_isolated" else "")
                all_rows.append({"segment": name, "layer": -1, "exec_style": style,
                                 "timing": timing, "variant": variant,
                                 "median_ms": round(s_["median_ms"], 6),
                                 "p5_ms": round(s_["p5_ms"], 6),
                                 "p95_ms": round(s_["p95_ms"], 6),
                                 "min_ms": round(s_["min_ms"], 6), "n": s_["n"],
                                 "n_kernels": nk, "bytes": "", "flops": "",
                                 "achieved_GBps": "", "achieved_TFLOPps": ""})
                mi.add_stats("t_step_extra", s_, "ms", s_["n"], "segment_summary.csv",
                             key=name, variant=variant, exec_style=style, timing=timing)
                if timing == "wall_launch" and nk > 0:
                    mi.add("t_launch_per_kernel", s_["median_ms"] / nk, "ms", "median",
                           s_["n"], "segment_summary.csv", key=name, variant=variant,
                           exec_style=style, timing="wall_launch")
            if nk > 0:
                mi.add("n_kernels_host_segment", nk, "count", "value", 1,
                       "segment_summary.csv", key=name,
                       variant=f"step_extra=1;exec_style={style}", exec_style=style)
            print(f"  {name:22s} {style:9s} {all_rows[-4]['median_ms']:.4f} ms kernels={nk}",
                  flush=True)
        rows.append({"model": "pi0", "segment": name, "order_in_layer": -1,
                     "layer_kind": "per_step", "source_ref": ref, "ops": ops,
                     "inputs": "", "outputs": "", "bytes_read": "", "bytes_written": "",
                     "flops": "", "flops_formula": "", "in_model_code": "yes",
                     "notes": "per-step extra, not part of a layer"})
    for style in styles:
        tot = sum(r["median_ms"] for r in all_rows if r["layer"] == -1
                  and r["exec_style"] == style and r["timing"] == "wall_isolated"
                  and "gap_ms=0" in r["variant"])
        if tot > 0:
            mi.add("t_step_extra", tot, "ms", "median", 3, "segment_summary.csv",
                   key="total", variant="step_extra=1", exec_style=style,
                   timing="wall_isolated")
    return rows


def finish(out, args, gate, policy, cfg, shipped, real, tok, seg_rows, all_rows, layer_rows,
           samples, mi, notes, styles, Pn, Sn):
    # capture the last expert layer's output, which final_norm_out_proj consumes
    last = policy.model.paligemma_with_expert.gemma_expert.model.layers[-1]
    holder, orig = {}, last.forward

    def cap(*a, **k):
        r = orig(*a, **k)
        holder.setdefault("h", r[0] if isinstance(r, tuple) else r)
        return r

    print("\n=== per-step extras ===", flush=True)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    last.forward = cap
    try:
        with torch.no_grad():
            policy.predict_action_chunk(batch)
    finally:
        last.forward = orig
    kmap = {}
    if args.kernels_csv and Path(args.kernels_csv).exists():
        for r in csv.DictReader(Path(args.kernels_csv).open()):
            kmap[(r["segment"], r["exec_style"])] = float(r["n_kernels"])
    seg_rows += measure_extras(policy, batch, args, mi, all_rows, notes, styles,
                               holder["h"].detach().clone(), kmap)

    with (out / "segments.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()))
        w.writeheader()
        w.writerows(seg_rows)
    with (out / "segment_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    if layer_rows:
        with (out / "host_layer_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
            w.writeheader()
            w.writerows(layer_rows)
    sample_csv = out / "raw" / "segment_samples.csv"
    with sample_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["segment", "exec_style", "timing", "variant",
                                          "iter", "ms"])
        w.writeheader()
        w.writerows(samples)
    # protocol §3.5: hash and secrets-scan the ORIGINAL, then zstd it
    orig_entry = compress_and_record(sample_csv)
    write_json(out / "raw" / "ORIGINALS.json", {
        "_purpose": "protocol §3.5/§6: committed compressed; records the compression command "
                    "and the original's sha256, size and pre-compression secrets scan.",
        "files": [orig_entry]})

    # median launch cost across eager segments
    launch = [r["median_ms"] / r["n_kernels"] for r in all_rows
              if r["exec_style"] == "eager" and r["timing"] == "wall_launch"
              and isinstance(r["n_kernels"], float) and r["n_kernels"] > 0]
    if launch:
        launch.sort()
        mi.add("t_launch_per_kernel", launch[len(launch) // 2], "ms", "median", len(launch),
               "segment_summary.csv", exec_style="eager", timing="wall_launch")

    n_mi = mi.write(out / "model_inputs.csv")
    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11a", "task": "T010",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_revision": P.REVISION, "model_repo": P.CHECKPOINT,
        "model_commit": P.REVISION, "checkpoint": P.CHECKPOINT,
        "precision": "bfloat16", "compile_mode": None, "batch": 1,
        "num_denoise_steps": args.steps, "action_horizon": getattr(cfg, "chunk_size", None),
        "prefix_tokens": Pn, "suffix_tokens": Sn, "token_counts": tok,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "exec_styles": styles,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "jetson_clocks": "applied before the run; GPU locked min==max",
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/host_segments.py --outdir {args.outdir} "
                   f"--warmup {args.warmup} --timed {args.timed}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`; T005 showed "
                              "sustained load does not move it on this device.",
        "model_inputs_rows": n_mi,
        "not_measured": notes,
        "raw_files": [],
        "notes": [
            "Segments are the DENOISING path: denoise_step -> paligemma_with_expert.forward "
            "with inputs_embeds=[None, suffix] -> gemma_expert.model.forward -> the decoder "
            "layer at pi_gemma.py:157-188. compute_layer_complete serves the joint "
            "prefix+suffix path and is NOT what a denoising step runs.",
            "embed_suffix returns adarms_cond=None for pi0 (modeling_pi0.py:734), so "
            "PiGemmaRMSNorm takes its unconditioned branch and the residuals are plain x+y. "
            "pi05 is expected to differ, which is why the branch is recorded per model.",
            "eager_attention_forward calls repeat_kv (modeling_gemma.py:205-206), expanding "
            "the single KV head to 8 before both matmuls. Unique KV bytes are counted once in "
            "`bytes_host_segment`; the expansion is noted as traffic the analytic model omits.",
            "Inputs are the model's own tensors, captured from one real denoising step of the "
            "fixture and replayed; every segment calls the model's own callable where one "
            "exists. `partials_residual` and `cast_scores_fp32_fp16` are mapping_only.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}  ({n_mi} model_inputs rows, {len(samples)} samples)")
    if notes:
        print(f"  {len(notes)} not-measured item(s):")
        for x in notes[:10]:
            print(f"    {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
