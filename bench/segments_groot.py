"""GR00T's DiT block, captured and split into segments comparable with π0's.

T012 needs E11's per-segment bytes and FLOPs for GR00T, the way T010 produced them for
π0. GR00T's action head is not a Gemma decoder at all — it is a DiT
(`gr00t/model/modules/dit.py`), and the differences are the point of the comparison, so
they are listed rather than smoothed over.

## The wrap point exists, unlike SmolVLA's

`DiT.forward` calls `block(...)` for each block (:310-326), and `block` is an
`nn.Module`, so `transformer_blocks[i].forward` **is** reached. π0's capture style works
here; SmolVLA's does not, and needed its own module. Checking this before writing
anything was the point — assuming it would have been the same mistake as assuming
`regions.py` worked for SmolVLA.

## Four structural differences from the Gemma expert

1. **The blocks alternate.** `DiT.forward` (:311) sends `encoder_hidden_states` only on
   even indices when `interleave_self_attention` is set, so odd blocks are pure
   self-attention and even blocks are cross-attention onto the VLM's tokens. π0's expert
   layers are all the same kind. The capture records which kind it saw.
2. **The conditioning is shift+scale with NO gate.** `AdaLayerNorm.forward` (:89-96) is
   `LayerNorm(x) * (1 + scale) + shift`, where `scale, shift = linear(silu(temb)).chunk(2)`.
   π0.5's AdaRMS returns a *gate* applied to the residual instead; π0 has neither. So the
   per-token conditioning traffic differs three ways across the three architectures.
3. **The norms are LayerNorm, not RMSNorm.** LayerNorm subtracts the mean as well as
   scaling by the variance, which is one more pass over the row.
4. **The attention is FUSED.** `attn1` is a diffusers `Attention` running under
   `_sdpa_context()`, so score-matmul, mask, softmax and context-matmul happen inside one
   kernel. π0's eager Gemma path exposes them as separate tensors and this file cannot,
   without replacing the model's own arithmetic. The `attention_fused` segment therefore
   covers all four, and the fact that they are not separable IS the measurement — it is
   reported, not worked around.
"""
from __future__ import annotations

import importlib
from contextlib import nullcontext

import torch


def _sdpa_ctx(block):
    """The model's OWN SDPA context manager, taken from the block's own module.

    `BasicTransformerBlock.forward` wraps `attn1` in `_sdpa_context()` (dit.py:197). On
    this device that returns `nullcontext()` — the forced-math path is for sm121 — so it
    changes nothing today. Using it anyway means the replay cannot silently diverge from
    the model if that check ever changes, which is the whole discipline of replaying with
    the model's own modules rather than an equivalent of them. Resolved through the
    block's `__module__` so the three GR00T worktrees each get their own.
    """
    try:
        mod = importlib.import_module(type(block).__module__)
        return getattr(mod, "_sdpa_context", nullcontext)()
    except Exception:                                          # noqa: BLE001
        return nullcontext()


SRC = "gr00t/model/modules/dit.py"


class DiTBlockCapture:
    """Record one DiT block's real arguments during a denoising step.

    Patches `transformer_blocks[idx].forward`, which `DiT.forward` genuinely calls.
    """

    def __init__(self, model, block_idx: int):
        self.blocks = model.action_head.model.transformer_blocks
        self.block = self.blocks[block_idx]
        self.idx = block_idx
        self.args = None
        self.kind = None
        self._orig = None

    def __enter__(self):
        cap = self
        self._orig = self.block.forward

        def wrapper(hidden_states, attention_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, temb=None, **kw):
            if cap.args is None:                     # first denoising step only
                cap.kind = ("cross_attn" if encoder_hidden_states is not None
                            else "self_attn")
                cap.args = {
                    "hidden_states": hidden_states.detach().clone(),
                    "attention_mask": None if attention_mask is None
                    else attention_mask.detach().clone(),
                    "encoder_hidden_states": None if encoder_hidden_states is None
                    else encoder_hidden_states.detach().clone(),
                    "encoder_attention_mask": None if encoder_attention_mask is None
                    else encoder_attention_mask.detach().clone(),
                    "temb": None if temb is None else temb.detach().clone(),
                }
            return cap._orig(hidden_states, attention_mask=attention_mask,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask,
                             temb=temb, **kw)

        self.block.forward = wrapper
        return self

    def __exit__(self, *exc):
        self.block.forward = self._orig
        return False


# Set once by the driver (bench/groot_segments.py) before any capture is
# reconstructed, so intermediates() keeps its single-argument signature.
_API = [None]


def model_autocast_ctx(api: str | None = None):
    """The autocast context this GR00T policy runs inference under, and a note.

    Returns (context, note). The decision is driven by `api`, the SAME condition
    bench/groot_phases.py:347 uses to time n1.5 correctly:

        if args.api == "old":
            # N1.5's own policy runs the model under inference_mode + autocast
            # (policy.py:189-191); timing it any other way would not be its inference.

    ## Why this is not source inspection any more

    The first version inspected `gr00t.model.policy` for the string "torch.autocast" and
    returned nullcontext inside `except (ImportError, OSError)`. It verified correctly when
    I tested it by hand, and then returned nullcontext in the real run and SAID NOTHING, so
    n1.5 failed a second time with the same dtype error one Linear earlier
    (`attn.to_q(hidden_states)` instead of `attn.to_k(encoder_hidden_states)`). A silent
    except-and-degrade is the wrong shape for a decision that changes what gets measured:
    whatever swallowed it, the run could not tell me.

    So the caller passes the flag it already has, and the source check is kept only as a
    CROSS-CHECK that raises on disagreement rather than as the mechanism. Both answers are
    printed, so a run states what it did instead of leaving it to be inferred.
    """
    want = (api == "old")
    seen = None
    try:
        import inspect
        pol = importlib.import_module("gr00t.model.policy")
        seen = "torch.autocast" in inspect.getsource(pol)
    except Exception as e:                      # noqa: BLE001 - reported, never swallowed
        print(f"  autocast cross-check unavailable ({type(e).__name__}: {e}); "
              f"relying on --api={api}", flush=True)

    if seen is not None and seen != want:
        raise SystemExit(
            f"autocast disagreement: --api={api} implies autocast={want}, but "
            f"gr00t.model.policy source says autocast={seen}. bench/groot_phases.py:347 "
            f"keys this on api=='old' and produced the published n1.5 latencies, so one of "
            f"the two premises has changed. Resolve it rather than guessing, because it "
            f"decides the dtype -- and therefore the BYTES -- of every segment.")

    if not want:
        print(f"  autocast: none (--api={api}); this model's policy.py does not wrap "
              f"inference in it", flush=True)
        return nullcontext(), ""

    dt = torch.bfloat16
    note = (f"Segments were reconstructed under torch.autocast(cuda, {dt}), because this "
            f"model's own gr00t/model/policy.py:189-191 wraps inference in it (--api=old). "
            f"The reported bytes are the traffic of the ops as autocast runs them; "
            f"autocast's OWN cast of the fp32 encoder states to {dt} is additional traffic "
            f"the shared bytes_formula does not model, and it does not arise for n1.7/n1.6, "
            f"whose policy.py has no autocast.")
    print(f"  autocast: torch.autocast(cuda, {dt}) (--api={api}, source cross-check "
          f"{'agrees' if seen else 'unavailable'})", flush=True)
    return torch.autocast(device_type="cuda", dtype=dt), note


def intermediates(cap: DiTBlockCapture) -> dict:
    """Reproduce the tensors the block passes between segments, in its own order.

    Follows BasicTransformerBlock.forward (:186-218) exactly, with the model's own
    modules. Raises if nothing was captured -- an empty capture must be loud.
    """
    if cap.args is None:
        raise SystemExit(
            f"DiT block {cap.idx} was never called. DiT.forward iterates every block, so "
            f"this means the denoising pass did not run at all. Refusing to build "
            f"segments from nothing.")
    b, a = cap.block, cap.args
    x = a["hidden_states"]
    ehs = a["encoder_hidden_states"]
    mask = a["encoder_attention_mask"] if ehs is not None else a["attention_mask"]

    ac, _ = model_autocast_ctx(_API[0])
    with torch.no_grad(), ac:
        if getattr(b, "norm_type", "") == "ada_norm":
            normed1 = b.norm1(x, a["temb"])                     # :190, shift+scale
            # the conditioning path, separately, because it is per-STEP work that a
            # memory-side model has to account for: one Linear on the timestep embedding
            # feeding every token's norm.
            cond = b.norm1.linear(b.norm1.silu(a["temb"]))      # :94
            scale, shift = cond.chunk(2, dim=1)                 # :95
        else:
            normed1 = b.norm1(x)
            cond = scale = shift = None
        pos_out = b.pos_embed(normed1) if b.pos_embed is not None else normed1  # :194-195

        with _sdpa_ctx(b):                                        # :197
            attn_out = b.attn1(pos_out, encoder_hidden_states=ehs, attention_mask=mask)
        if getattr(b, "final_dropout", None) is not None:
            attn_out = b.final_dropout(attn_out)                # :204-205, eval no-op
        resid1 = attn_out + x                                   # :207, PLAIN add
        if resid1.ndim == 4:
            resid1 = resid1.squeeze(1)
        normed3 = b.norm3(resid1)                               # :212, LayerNorm
        ff_out = b.ff(normed3)                                  # :213
        resid2 = ff_out + resid1                                # :215

    return {"x": x, "normed1": normed1, "cond": cond, "scale": scale, "shift": shift,
            "pos_out": pos_out, "attn_out": attn_out, "resid1": resid1,
            "normed3": normed3, "ff_out": ff_out, "resid2": resid2,
            "encoder_hidden_states": ehs, "mask": mask, "temb": a["temb"],
            "block": b, "kind": cap.kind, "idx": cap.idx}


def _attn_with_ctx(block, hidden, ehs, mask):
    """attn1 under the model's own SDPA context, as BasicTransformerBlock.forward runs it."""
    with _sdpa_ctx(block):
        return block.attn1(hidden, encoder_hidden_states=ehs, attention_mask=mask)


def build_segments(t: dict, S: int, D: int, I: int, H: int, HD: int, T_keys: int
                   ) -> list[dict]:
    """GR00T's separable segments, named to line up with π0's where they correspond."""
    b = t["block"]
    B2 = 2                                             # bf16
    ehs = t["encoder_hidden_states"]
    Tk = ehs.shape[1] if ehs is not None else S        # cross-attends S_vlm, self-attends S
    cond_dim = t["cond"].shape[-1] if t["cond"] is not None else 0

    segs = []
    if t["cond"] is not None:
        segs.append(dict(
            segment="adaln_cond", order=0, in_model="yes",
            source_ref=f"{SRC}:94-95 (AdaLayerNorm.linear on silu(temb))",
            ops="linear(silu(temb)) then chunk into scale and shift",
            fn=lambda: b.norm1.linear(b.norm1.silu(t["temb"])).chunk(2, dim=1),
            inputs=f"temb:{tuple(t['temb'].shape)}",
            outputs=f"cond:{tuple(t['cond'].shape)}",
            bytes_formula="read temb[D] + weight[D, 2D]; write cond[2D], all bf16",
            bytes_read=(D + D * cond_dim) * B2, bytes_written=cond_dim * B2,
            flops=2 * D * cond_dim + 4 * D,
            flops_formula="2*D*2D (linear) + 4*D (silu)",
            notes="per-STEP, not per-token: one Linear on the timestep embedding whose "
                  "output conditions every token's norm. pi0 has no equivalent at all; "
                  "pi0.5's AdaRMS produces a GATE as well as a shift and scale"))
    segs.append(dict(
        segment="adalayernorm1", order=1, in_model="yes",
        source_ref=f"{SRC}:96 (AdaLayerNorm) or :192 (plain LayerNorm)",
        ops="LayerNorm(x) * (1 + scale) + shift — no gate",
        fn=(lambda: b.norm1(t["x"], t["temb"])) if t["cond"] is not None
        else (lambda: b.norm1(t["x"])),
        inputs=f"x:{tuple(t['x'].shape)}",
        outputs=f"normed:{tuple(t['normed1'].shape)}",
        bytes_formula="read x[S,D] + scale[D] + shift[D]; write normed[S,D], all bf16",
        bytes_read=S * D * B2 + 2 * D * B2, bytes_written=S * D * B2,
        flops=11 * S * D,
        flops_formula="9*S*D (LayerNorm: mean AND variance) + 2*S*D (scale, shift)",
        notes="LayerNorm subtracts the mean as well as scaling by the variance, so it is "
              "one more pass over each row than pi0's RMSNorm"))
    if b.pos_embed is not None:
        segs.append(dict(
            segment="pos_embed", order=2, in_model="yes",
            source_ref=f"{SRC}:194-195",
            ops="additive sinusoidal positional embedding on the normed states",
            fn=lambda: b.pos_embed(t["normed1"]),
            inputs=f"normed:{tuple(t['normed1'].shape)}",
            outputs=f"out:{tuple(t['pos_out'].shape)}",
            bytes_formula="read normed[S,D] + table[S,D]; write out[S,D], all bf16",
            bytes_read=2 * S * D * B2, bytes_written=S * D * B2,
            flops=S * D, flops_formula="S*D (one add per element)",
            notes="applied INSIDE the block, per block, not once at the sequence head as "
                  "RoPE is in the LeRobot models"))
    segs.append(dict(
        segment="attention_fused", order=3, in_model="yes",
        source_ref=f"{SRC}:197-203 (diffusers Attention under _sdpa_context)",
        ops=f"{t['kind']}: q/k/v projections, scores, softmax and context in ONE fused "
            f"SDPA kernel",
        fn=lambda: _attn_with_ctx(b, t["pos_out"], ehs, t["mask"]),
        inputs=f"q_in:{tuple(t['pos_out'].shape)}"
               + (f";kv_in:{tuple(ehs.shape)}" if ehs is not None else ""),
        outputs=f"attn:{tuple(t['attn_out'].shape)}",
        bytes_formula="read the query states and the key/value source; write the context",
        bytes_read=(S * D + (Tk * D if ehs is not None else 0)) * B2,
        bytes_written=S * D * B2,
        flops=4 * H * S * Tk * HD,
        flops_formula="2*H*S*T*HD (scores) + 2*H*S*T*HD (context), projections excluded",
        notes="NOT separable into score / softmax / context. SDPA fuses them into one "
              "kernel, so splitting them here would mean replacing the model's own "
              "arithmetic with a different implementation and measuring that instead. "
              "pi0's eager Gemma path DOES expose them separately, which is why its "
              "segment table has a `softmax` row and this one does not. The difference "
              "is the measurement, not a gap in it."))
    segs.append(dict(
        segment="residual", order=4, in_model="yes",
        source_ref=f"{SRC}:207",
        ops="attn_output + hidden_states (plain add, no gate)",
        fn=lambda: t["attn_out"] + t["x"],
        inputs=f"x:{tuple(t['x'].shape)};y:{tuple(t['attn_out'].shape)}",
        outputs=f"out:{tuple(t['resid1'].shape)}",
        bytes_formula="read x[S,D] + y[S,D]; write out[S,D], all bf16",
        bytes_read=2 * S * D * B2, bytes_written=S * D * B2,
        flops=S * D, flops_formula="S*D",
        notes="GR00T gates the NORM, not the residual; pi0.5 gates the residual"))
    segs.append(dict(
        segment="layernorm3", order=5, in_model="yes",
        source_ref=f"{SRC}:212",
        ops="plain LayerNorm before the feed-forward",
        fn=lambda: b.norm3(t["resid1"]),
        inputs=f"x:{tuple(t['resid1'].shape)}",
        outputs=f"normed3:{tuple(t['normed3'].shape)}",
        bytes_formula="read x[S,D]; write normed[S,D], all bf16",
        bytes_read=S * D * B2, bytes_written=S * D * B2,
        flops=9 * S * D, flops_formula="9*S*D (LayerNorm)",
        notes="unconditioned, unlike norm1"))
    segs.append(dict(
        segment="feedforward", order=6, in_model="yes",
        source_ref=f"{SRC}:213 (diffusers FeedForward, activation from the config)",
        ops="the whole FF block: up projection, activation, down projection",
        fn=lambda: b.ff(t["normed3"]),
        inputs=f"normed3:{tuple(t['normed3'].shape)}",
        outputs=f"ff:{tuple(t['ff_out'].shape)}",
        bytes_formula="read normed[S,D] + the FF weights; write ff[S,D], all bf16",
        bytes_read=S * D * B2 + 2 * D * I * B2, bytes_written=S * D * B2,
        flops=4 * S * D * I + 9 * S * I,
        flops_formula="2*S*D*I (up) + 2*S*I*D (down) + 9*S*I (activation)",
        notes="reported whole. pi0's table splits act_fn(gate)*up out from its "
              "projections per protocol §13.2; the diffusers FeedForward is a Sequential "
              "whose pieces would have to be indexed by position, which is brittle "
              "against a version change. Split later if the PIM model needs it"))
    segs.append(dict(
        segment="residual_ff", order=7, in_model="yes",
        source_ref=f"{SRC}:215",
        ops="ff_output + hidden_states (plain add)",
        fn=lambda: t["ff_out"] + t["resid1"],
        inputs=f"x:{tuple(t['resid1'].shape)};y:{tuple(t['ff_out'].shape)}",
        outputs=f"out:{tuple(t['resid2'].shape)}",
        bytes_formula="read x[S,D] + y[S,D]; write out[S,D], all bf16",
        bytes_read=2 * S * D * B2, bytes_written=S * D * B2,
        flops=S * D, flops_formula="S*D",
        notes=""))

    # EVERY segment closure must run under the same autocast the real forward runs under,
    # not just the attention one.
    #
    # Wrapping only intermediates() got n1.5 past the reconstruction and then it died in the
    # TIMING closures, at the same dtype error from a different call site
    # (`block.attn1(hidden, ...)` in _attn_with_ctx rather than line 203). Patching
    # _attn_with_ctx alone would have been the same mistake a third time: the other closures
    # run Linears, LayerNorms and a FeedForward too. Those happen not to crash, because
    # their captured inputs are already dtype-consistent -- but without autocast they time
    # ops WITHOUT the cast the model inserts, which is a wrong per-segment time rather than
    # a crash. A silently wrong number is the worse of the two outcomes.
    #
    # Wrapped here, once, over the whole list, so a segment added later cannot miss it.
    ac, _ = model_autocast_ctx(_API[0])
    if not isinstance(ac, nullcontext):
        def _under_autocast(fn):
            def wrapped():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return fn()
            return wrapped
        for sd in segs:
            sd["fn"] = _under_autocast(sd["fn"])

    return segs
