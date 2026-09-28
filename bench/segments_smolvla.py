"""SmolVLA's expert layer, captured and split into the same segments as π0's.

`bench/host_segments.py` measures one expert layer of π0 segment by segment, replaying
each piece with the model's own modules so the bytes and FLOPs are the real ones. It
cannot do that for SmolVLA, for a reason that is worth stating precisely rather than
calling it "unsupported":

**`SmolVLMWithExpertModel.forward` inlines the layer body.** It never calls
`layers[j].forward`. It calls `forward_attn_layer` (:197) or `forward_cross_attn_layer`
(:274) to get the attention output, and then applies `self_attn.o_proj`,
`post_attention_layernorm` and `mlp` itself at :472-480. So patching `layers[j].forward`
— which is how π0's capture works — yields an **empty** capture, silently, and every
number downstream would be computed from a zero tensor and look entirely plausible.
`bench/model_registry.py` records this as `layer_capture="patch_container_attn_fns"` and
`LayerCapture` refuses the model outright rather than returning nothing.

The wrap point that DOES see a layer's real arguments is those two methods, which is what
this module patches.

## Five differences from the Gemma path that change the PIM inputs

Read from `smolvlm_with_expert.py` at the installed commit, not assumed:

1. **The attention scores are computed in fp32.** `eager_attention_forward` (:528-537)
   upcasts q and k to float32 *before* the matmul, so the score matmul is an fp32 GEMM,
   not a bf16 one. π0's is bf16 with an fp32 softmax. That is a different arithmetic
   intensity and a different energy per score.
2. **The mask is applied with `torch.where`, not an add** (:539-540). It selects between
   the score and `finfo(float32).min` rather than adding a large negative number, so it
   reads a boolean mask and writes a full fp32 matrix instead of doing H·S·T adds.
3. **Tensors stay in B, L, H, D.** There is no `transpose(1, 2)` after the projections
   (:225-228); the transpose happens inside the attention. Layouts decide coalescing, so
   a memory-side model cannot inherit π0's.
4. **RoPE runs in fp32 and is SmolVLA's own** (`apply_rope`, :28-50), not
   `transformers`' `apply_rotary_pos_emb`. It upcasts, builds the timescale table on the
   fly, and casts back.
5. **The residuals are plain adds** (`out_emb += hidden_states`, :474 and :480) with no
   gate, unlike π0.5's AdaRMS gating. The norms take no `cond`.

## The KV cache is a dict, not a Cache object

π0's layer receives a `transformers` Cache and reads `past_key_values.layers[i].keys`.
SmolVLA keeps `past_key_values[layer_idx] = {"key_states": ..., "value_states": ...}` and
concatenates on **dim=1** (:263-264), because of difference 3. The capture reads it that
way; reading it π0's way would raise, which is the good failure, but it is easier to just
be right.
"""
from __future__ import annotations

import torch

SRC = "lerobot/policies/smolvla/smolvlm_with_expert.py"


def _iadd(a, b):
    """`a += b`, on a copy. The model's residuals are in-place (:474, :480) and that is
    not a stylistic detail: o_proj's output is bf16 and the hidden states are fp32, so an
    out-of-place add promotes and the following bf16 matmul fails. Timing `a + b` here
    would also be timing a wider add than the model performs."""
    out = a.clone()
    out += b
    return out


class SmolVLALayerCapture:
    """Record one expert layer's real arguments during a denoising step.

    `layer_idx` is the **VLM** layer index, which is what `forward` iterates over.
    `get_model_layers` (:388-401) maps it to an expert layer through
    `i // (num_vlm_layers // num_expert_layers)`, and yields `None` for indices that do
    not map. The capture records the mapping it actually saw, so a configuration where
    the two counts differ cannot be silently misread as one where they do not.
    """

    def __init__(self, policy, layer_idx: int, spec):
        self.container = getattr(policy.model, spec.container)
        self.idx = layer_idx
        self.args = None
        self.kind = None
        self.att_output = None
        self.replay = None
        self._orig = {}

    def __enter__(self):
        cap = self
        c = self.container

        def wrap(name, kind):
            orig = getattr(c, name)

            def wrapper(model_layers, inputs_embeds, layer_idx, position_ids,
                        attention_mask, batch_size, head_dim, use_cache=True,
                        fill_kv_cache=True, past_key_values=None, **kw):
                # Only the FIRST denoising step, and only the requested layer. During
                # denoising inputs_embeds is [None, suffix], so index 1 is the expert.
                if cap.args is None and layer_idx == cap.idx and not fill_kv_cache:
                    layer = model_layers[1][layer_idx]
                    hs = inputs_embeds[1]
                    if layer is not None and hs is not None:
                        pk = pv = None
                        if past_key_values is not None and layer_idx in past_key_values:
                            e = past_key_values[layer_idx]
                            pk = e["key_states"].detach().clone()
                            pv = e["value_states"].detach().clone()
                        cap.kind = kind
                        cap.args = {
                            "hidden_states": hs.detach().clone(),
                            "attention_mask": None if attention_mask is None
                            else attention_mask.detach().clone(),
                            "position_ids": None if position_ids is None
                            else position_ids.detach().clone(),
                            "batch_size": batch_size, "head_dim": head_dim,
                            # the PREFIX K/V for this layer, read BEFORE the attention
                            # concatenates this step's keys onto it, so it holds the P
                            # prefix tokens only. Getting this wrong understates T by the
                            # whole prefix, which is the single largest error available
                            # here (it was a 17x understatement for pi0).
                            "prefix_k": pk, "prefix_v": pv,
                            "layer": layer,
                        }
                        # 2026-09-28: the call itself, so `attention_whole` can REPLAY the model's
                        # own attention. Before this fix the segment returned the captured output
                        # tensor (fn=lambda: ctx) and so timed a Python return, not attention.
                        # With fill_kv_cache=False the call only reads the cache (:271-276 concat),
                        # so replaying it on a cloned one-layer cache is side-effect free.
                        kv = None if pk is None else {layer_idx: {"key_states": pk, "value_states": pv}}
                        emb = [None, hs.detach().clone()]
                        cap.replay = lambda _o=orig, _ml=model_layers, _e=emb, _li=layer_idx, \
                            _pos=cap.args["position_ids"], _m=cap.args["attention_mask"], \
                            _b=batch_size, _h=head_dim, _u=use_cache, _kv=kv, _kw=dict(kw): \
                            _o(_ml, _e, _li, _pos, _m, _b, _h, use_cache=_u, fill_kv_cache=False,
                               past_key_values=_kv, **_kw)
                out = orig(model_layers, inputs_embeds, layer_idx, position_ids,
                           attention_mask, batch_size, head_dim, use_cache=use_cache,
                           fill_kv_cache=fill_kv_cache, past_key_values=past_key_values,
                           **kw)
                # The attention OUTPUT, as the model produced it. SmolVLA's two layer
                # kinds wire their projections differently -- a cross-attention layer's
                # k_proj/v_proj take the VLM's key width and are applied to the CACHED
                # VLM keys (:352-360), while a self-attention layer's take the expert's
                # and are applied to its own states. Reconstructing q/k/v here therefore
                # needs a different computation per kind, and three attempts at it each
                # found another asymmetry. Taking the output instead is exact for every
                # kind and cannot drift.
                if cap.args is not None and cap.att_output is None:
                    try:
                        o = out[0][0] if isinstance(out, tuple) else out[0]
                        cap.att_output = o.detach().clone()
                    except (IndexError, TypeError, AttributeError):
                        pass
                return out

            self._orig[name] = orig
            setattr(c, name, wrapper)

        wrap("forward_attn_layer", "self_attn")
        wrap("forward_cross_attn_layer", "cross_attn")
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self.container, name, orig)
        return False


def intermediates(policy, cap: SmolVLALayerCapture, spec) -> dict:
    """Reproduce every tensor the layer body passes between segments.

    Follows `forward_attn_layer` (:213-270) and then `forward`'s inlined tail (:466-481),
    in that order, using the model's own modules. Raises rather than guessing if the
    capture is empty — an empty capture here used to be the failure mode, and it must be
    loud.
    """
    if cap.args is None:
        raise SystemExit(
            f"layer {cap.idx} was never captured. Either the layer index does not map to "
            f"an expert layer (get_model_layers returns None for unmapped indices), or "
            f"the denoising pass did not run. Refusing to build segments from nothing.")
    a = cap.args
    layer = a["layer"]
    x = a["hidden_states"]
    attn = layer.self_attn
    c = getattr(policy.model, spec.container)
    bsz, hd = a["batch_size"], a["head_dim"]

    if cap.att_output is None:
        raise SystemExit(
            f"layer {cap.idx}: the attention output was never captured. The wrapper saw "
            f"the call but could not read its return value -- refusing to build segments "
            f"from a reconstruction.")

    with torch.no_grad():
        normed1 = layer.input_layernorm(x)                             # :216, no cond
        # The attention as the MODEL computed it. q/k/v are deliberately NOT rebuilt:
        # a cross-attention layer projects the CACHED VLM keys through the expert's
        # k_proj/v_proj (:352-360), which are Linears sized for the VLM's key width
        # (:113-122), while a self-attention layer projects the expert's own states.
        # The two kinds are different computations, so `rope` and `softmax` are not
        # separable for this model without reimplementing both -- which is reported as
        # such rather than approximated. GR00T's fused SDPA gets the same treatment.
        ctx = cap.att_output
        start = 0
        end = start + x.shape[1]
        att_slice = ctx[:, start:end] if ctx.shape[1] >= end else ctx     # :471

        # ---- the inlined tail in forward (:466-481) ------------------------------------
        att_out = att_slice
        if att_out.dtype != attn.o_proj.weight.dtype:                  # :469-470
            att_out = att_out.to(attn.o_proj.weight.dtype)
        attn_out = attn.o_proj(att_out)                                # :472
        # IN-PLACE, exactly as the model does it: `out_emb += hidden_states` (:474).
        # That is not the same as `attn_out + x`. o_proj's output is bf16 and the
        # hidden states are fp32, so the out-of-place add PROMOTES to fp32 and the very
        # next line died with
        #     RuntimeError: expected mat1 and mat2 to have the same dtype,
        #                   but got: float != c10::BFloat16
        # at mlp.gate_proj, whose weights are bf16. An in-place add keeps the target's
        # dtype, which is why the model itself does not hit this. Replaying with `+`
        # where the model writes `+=` is replaying a different computation.
        resid1 = attn_out.clone()
        resid1 += x                                                    # :474
        normed2 = layer.post_attention_layernorm(resid1)               # :477, no cond
        g = layer.mlp.gate_proj(normed2)
        u = layer.mlp.up_proj(normed2)
        mlp_out = layer.mlp.down_proj(layer.mlp.act_fn(g) * u)
        resid2 = mlp_out.clone()
        resid2 += resid1                                               # :480

    return {"x": x, "normed1": normed1, "gate1": None,
            "ctx": ctx, "att_slice": att_slice, "attn_out": attn_out, "resid1": resid1,
            "normed2": normed2, "gate2": None, "g": g, "u": u,
            "mlp_out": mlp_out, "resid2": resid2,
            "attn": attn, "layer": layer, "adarms_cond": None,
            "container": c, "kind": cap.kind, "bsz": bsz, "head_dim": hd,
            "attn_replay": cap.replay,
            "mask_full": a["attention_mask"],
            # not separable for this model -- see the note in intermediates
            "not_separable": ("rope", "softmax", "attention_gpu")}


def build_segments(t: dict, S: int, D: int, I: int, H: int, HD: int, T_keys: int
                   ) -> list[dict]:
    """The same segment names as π0's, so the cross-model table lines up.

    Where SmolVLA does the work differently the `ops`, `source_ref`, byte and FLOP fields
    say so — the point of the comparison is the difference, so it must not be smoothed
    over by reusing π0's numbers.
    """
    layer, attn = t["layer"], t["attn"]
    B2, B4 = 2, 4                                          # bf16, fp32
    bsz, hd = t["bsz"], t["head_dim"]
    # `mask` was only used by the score/softmax segments, which this model does not
    # decompose (see intermediates). Left here as a local so the signature of the
    # remaining segments is unchanged, and read with .get so a future tensor removal
    # fails at ITS use site rather than at the top of the builder.
    mask = t.get("mask_full")

    return [
        dict(segment="rmsnorm1", order=0, in_model="yes",
             source_ref=f"{SRC}:216",
             ops="input_layernorm(x) — plain RMSNorm, NO cond (SmolVLA has no AdaRMS)",
             fn=lambda: layer.input_layernorm(t["x"]),
             inputs=f"x:{tuple(t['x'].shape)}",
             outputs=f"normed:{tuple(t['normed1'].shape)}",
             bytes_formula="read x[S,D] bf16 + weight[D] bf16; write normed[S,D] bf16",
             bytes_read=S * D * B2 + D * B2, bytes_written=S * D * B2,
             flops=8 * S * D, flops_formula="8*S*D (rmsnorm)",
             notes="no conditioning tensor, unlike pi0.5"),
        dict(segment="residual_rmsnorm2", order=3, in_model="yes",
             source_ref=f"{SRC}:474-477",
             ops="out_emb += x IN-PLACE (no gate); then post_attention_layernorm",
             fn=lambda: layer.post_attention_layernorm(_iadd(t["attn_out"], t["x"])),
             inputs=f"x:{tuple(t['x'].shape)};attn_out:{tuple(t['attn_out'].shape)}",
             outputs=f"normed2:{tuple(t['normed2'].shape)}",
             bytes_formula="read x + attn_out + weight; write the sum and the normed result",
             bytes_read=3 * S * D * B2, bytes_written=2 * S * D * B2,
             flops=9 * S * D, flops_formula="S*D (add) + 8*S*D (rmsnorm)",
             notes="SmolVLA's residual is a plain +=; there is no AdaRMS gate to apply"),
        dict(segment="gelu_gate_mul_up", order=4, in_model="yes",
             source_ref=f"{SRC}:478 (mlp), act_fn from the expert's own config",
             ops="act_fn(gate_proj_out) * up_proj_out",
             fn=lambda: layer.mlp.act_fn(t["g"]) * t["u"],
             inputs=f"gate:{tuple(t['g'].shape)}:{t['g'].dtype};up:{tuple(t['u'].shape)}",
             outputs=f"h:{tuple(t['g'].shape)}",
             bytes_formula="read gate[S,I] + up[S,I]; write h[S,I], all bf16",
             bytes_read=2 * S * I * B2, bytes_written=S * I * B2,
             flops=9 * S * I, flops_formula="9*S*I (activation ~8 flops + 1 multiply)",
             notes="the three projections around it are large linears and are NOT part of "
                   "the segment (protocol §13.2)"),
        dict(segment="residual", order=6, in_model="yes",
             source_ref=f"{SRC}:480",
             ops="out_emb += after_first_residual, in-place",
             fn=lambda: _iadd(t["mlp_out"], t["resid1"]),
             inputs=f"x:{tuple(t['resid1'].shape)};y:{tuple(t['mlp_out'].shape)}",
             outputs=f"out:{tuple(t['resid1'].shape)}",
             bytes_formula="read x[S,D] + y[S,D]; write out[S,D], all bf16",
             bytes_read=2 * S * D * B2, bytes_written=S * D * B2,
             flops=S * D, flops_formula="S*D",
             notes="no gate, so this is one add per element"),
        dict(segment="attention_whole", order=7, in_model="yes",
             source_ref=f"{SRC}:197 / :274 (forward_attn_layer / forward_cross_attn_layer)",
             ops=f"{t['kind']}: the model's own forward_{'cross_' if t['kind'] == 'cross_attn' else ''}attn_layer "
                 "replayed on the captured inputs -- q/k/v projections + RoPE + scores + softmax + context "
                 "(o_proj is outside, in forward's inlined tail)",
             fn=t["attn_replay"],
             inputs=f"captured from the model's own call",
             outputs=f"att:{tuple(t['ctx'].shape)}",
             bytes_formula="not decomposed; see notes",
             bytes_read=0, bytes_written=t["ctx"].numel() * B2,
             flops=0, flops_formula="not decomposed; see notes",
             notes="NOT separable into rope / score / softmax / context for this model. "
                   "A cross-attention layer projects the CACHED VLM keys through the "
                   "expert's k_proj/v_proj, which are Linears sized for the VLM's key "
                   "width (:113-122, :352-360), while a self-attention layer projects "
                   "the expert's own states. The two kinds are different computations, "
                   "so reconstructing the pieces means reimplementing both -- and three "
                   "attempts at it each uncovered another asymmetry. The output is taken "
                   "as measured; the fact that it is not decomposable IS the measurement, "
                   "as it is for GR00T's fused SDPA."),
    ]
