"""Protocol §2 region instrumentation by wrapping (never editing), for the LeRobot family and GR00T.
Ported from reference/orin/bench/regions.py (LeRobot) and the timing hooks of reference/orin/bench/groot_phases.py,
with GR00T's vision and LLM calls wrapped SEPARATELY (the task's step 2; on the Orin vlm_prefix was derived).

Every region emits an NVTX range (+ a bracket-free alias for ncu) and a CUDA-event pair, read after the iteration's
final synchronise. Install AFTER any compilation (compile first, instrument second).

LeRobot (pi0, pi05, smolvla; container = paligemma_with_expert or vlm_with_expert):
  vision      <container>.embed_image            once per camera, summed
  vlm_prefix  <container>.forward, prefill branch (inputs_embeds[1] is None)
  ae_step[i]  model.denoise_step                 one per denoising step
  ae_total    derived: first ae_step start -> last ae_step end
GR00T (N1.7 Qwen3-VL backbone; N1.6 / N1.5 Eagle backbone):
  vlm_backbone  backbone.forward                 the enclosing call (validation: vision + vlm_prefix vs it)
  vision        Qwen3-VL: backbone.model.model.get_image_features  (visual tower incl. its merger)
                Eagle:    <eagle model>.extract_feature            (vision_model + pixel shuffle + mlp1 projector)
  vlm_prefix    the LLM decoder stack: Eagle eagle.language_model.model.forward; Qwen3-VL inner.language_model.forward
  backbone_lm_head  the vocabulary lm_head (computed by the shipped code, never used by GR00T)
  ae_total      action_head.get_action
  ae_step[i]    from action_head.action_encoder call i to call i+1 (the last one to the end of get_action): the
                whole loop body (encoder, DiT, decoder, Euler update)
  ae_dit[i]     action_head.model.forward (the DiT alone)

Accounting mode (Recorder(account=True)): every region also collects, for one eager pass,
  flops        torch.utils.flop_counter.FlopCounterMode, nested per region (inclusive)
  param_bytes  parameters of every module called while the region is open (nn.Embedding: rows looked up only)
  kv_bytes     LeRobot: bytes of the past_key_values the prefill returns; GR00T: encoder_hidden_states per DiT call
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch


def _register_flash_attn_flops():
    """FlopCounterMode has formulas for aten's attention ops, not for the flash_attn package's custom ops, which the
    GR00T backbones call (flash_attention_2). Register them so their attention FLOPs are counted:
      fixed length  q [B, Sq, H, D], k [B, Sk, Hk, D]: 4 * B * H * D * Sq * Sk (causal, Sq == Sk: Sq (Sq + 1) / 2 pairs)
      varlen        q [total_q, H, D], cu_seqlens_q / _k: the same summed over the packed sequences"""
    try:
        import flash_attn  # noqa: F401
        from torch.utils.flop_counter import register_flop_formula
        ops = torch.ops.flash_attn
    except Exception:
        return False

    def pairs(sq, sk, causal):
        return sq * (sq + 1) // 2 if causal and sq == sk else sq * sk

    @register_flop_formula(ops._flash_attn_forward, get_raw=True)
    def _fa(q, k, v, dropout_p, softmax_scale, causal, *args, out_val=None, **kw):
        b, sq, h, d = q.shape
        return 4 * b * h * d * pairs(sq, k.shape[1], causal)

    @register_flop_formula(ops._flash_attn_varlen_forward, get_raw=True)
    def _fav(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, softmax_scale, causal, *args, out_val=None, **kw):
        h, d = q.shape[1], q.shape[2]
        lq, lk = cu_q.tolist(), cu_k.tolist()
        return sum(4 * h * d * pairs(lq[i + 1] - lq[i], lk[i + 1] - lk[i], causal) for i in range(len(lq) - 1))
    return True


FLASH_ATTN_FLOPS_REGISTERED = _register_flash_attn_flops()


def tensor_bytes(obj, seen=None) -> int:
    """Bytes of every distinct tensor reachable through lists / tuples / dicts / cache objects."""
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        if obj.data_ptr() in seen:
            return 0
        seen.add(obj.data_ptr())
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(o, seen) for o in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(o, seen) for o in obj.values())
    total = 0
    for attr in ("key_cache", "value_cache", "layers", "keys", "values"):
        if hasattr(obj, attr):
            total += tensor_bytes(getattr(obj, attr), seen)
    return total


class Recorder:
    """Collects (region, start_event, end_event) spans for one iteration; optional accounting."""

    def __init__(self, account: bool = False):
        self.spans: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.step_idx = -1
        self._active = False
        self.account = account
        self.stack: list[str] = []
        self.flops: dict[str, int] = {}
        self.flop_ops: dict[str, dict] = {}
        self.params: dict[str, dict[int, int]] = {}
        self.kv_bytes: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self._open: dict[str, tuple] = {}
        self._hook = None

    def reset(self):
        self.spans.clear()
        self.step_idx = -1
        self._open.clear()

    def start(self):
        self._active = True
        if self.account and self._hook is None:
            self._hook = torch.nn.modules.module.register_module_forward_pre_hook(self._param_hook)

    def stop(self):
        self._active = False
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    # ------------------------------------------------------------------ accounting
    def _param_hook(self, module, args):
        if not self.stack:
            return
        if isinstance(module, torch.nn.Embedding):
            idx = args[0] if args else None
            nb = (idx.numel() * module.embedding_dim * module.weight.element_size()) if torch.is_tensor(idx) else 0
            for r in self.stack:
                d = self.params.setdefault(r, {})
                key = ("emb", id(module))
                prev = d.get(key, (0, str(module.weight.dtype)))[0]
                d[key] = (prev + nb, str(module.weight.dtype))
            return
        if type(module).__name__ == "CategorySpecificLinear" and hasattr(module, "num_categories"):
            # GR00T's per-embodiment Linear stores W [num_categories, in, out] and b [num_categories, out] but reads only
            # W[cat_ids] / b[cat_ids]: count the slices of the categories actually used (T004 review), not all of them
            cat = args[1] if len(args) > 1 else None
            ncat = int(torch.unique(cat).numel()) if torch.is_tensor(cat) else 1
            own = [(id(p), (p.numel() * p.element_size() * ncat // module.num_categories, str(p.dtype)))
                   for p in module.parameters(recurse=False)]
        else:
            own = [(id(p), (p.numel() * p.element_size(), str(p.dtype))) for p in module.parameters(recurse=False)]
        for r in self.stack:
            d = self.params.setdefault(r, {})
            for k, b in own:
                d[k] = b

    def _enter(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1
        if not self.account:
            return None
        from torch.utils.flop_counter import FlopCounterMode
        self.stack.append(name)
        m = FlopCounterMode(display=False)
        m.__enter__()
        return m

    def _exit(self, name, m):
        if m is None:
            return
        m.__exit__(None, None, None)
        self.flops[name] = self.flops.get(name, 0) + m.get_total_flops()
        ops = self.flop_ops.setdefault(name, {})
        for op, c in m.get_flop_counts().get("Global", {}).items():
            ops[str(op)] = ops.get(str(op), 0) + int(c)
        if name in self.stack:
            self.stack.remove(name)

    @contextmanager
    def region(self, name: str, alias: str | None = None):
        if not self._active:
            yield
            return
        torch.cuda.nvtx.range_push(name)
        if alias:
            torch.cuda.nvtx.range_push(alias)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        m = self._enter(name)
        s.record()
        try:
            yield
        finally:
            e.record()
            self._exit(name, m)
            self.spans.append((name, s, e))
            if alias:
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

    def begin(self, name: str, alias: str | None = None):
        """Open a region that does not follow one call (GR00T ae_step[i]); close it with end()."""
        if not self._active:
            return
        torch.cuda.nvtx.range_push(name)
        if alias:
            torch.cuda.nvtx.range_push(alias)
        s = torch.cuda.Event(enable_timing=True)
        m = self._enter(name)
        s.record()
        self._open[name] = (s, m, alias)

    def end(self, name: str):
        if name not in self._open:
            return
        s, m, alias = self._open.pop(name)
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._exit(name, m)
        self.spans.append((name, s, e))
        if alias:
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    # ------------------------------------------------------------------ results
    def results_ms(self, derive_ae_total: bool) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, s, e in self.spans:
            out[name] = out.get(name, 0.0) + s.elapsed_time(e)
        steps = [(n, s, e) for n, s, e in self.spans if n.startswith("ae_step[")]
        if steps and derive_ae_total:
            out["ae_total"] = steps[0][1].elapsed_time(steps[-1][2])
        return out

    def step_times_ms(self) -> list[float]:
        return [s.elapsed_time(e) for n, s, e in self.spans if n.startswith("ae_step[")]

    def accounting(self) -> dict:
        by_dtype = {}
        for r, d in self.params.items():
            t = {}
            for b, dt in d.values():
                t[dt] = t.get(dt, 0) + b
            by_dtype[r] = t
        return {"flops": dict(self.flops), "flop_ops": dict(self.flop_ops),
                "flash_attn_flop_formulas_registered": FLASH_ATTN_FLOPS_REGISTERED,
                "kv_bytes": dict(self.kv_bytes), "calls": dict(self.counts),
                "param_bytes": {r: sum(t.values()) for r, t in by_dtype.items()}, "param_bytes_by_dtype": by_dtype}


# ====================================================================== LeRobot
def instrument_lerobot(policy, rec: Recorder, container: str):
    model = policy.model
    pg = getattr(model, container)
    undo = []
    orig_embed_image = pg.embed_image

    def embed_image(self, image, _orig=orig_embed_image):
        with rec.region("vision"):
            return _orig(image)

    pg.embed_image = types.MethodType(embed_image, pg)
    undo.append(lambda: setattr(pg, "embed_image", orig_embed_image))

    orig_forward = pg.forward

    def forward(self, *a, _orig=orig_forward, **kw):
        embeds = kw.get("inputs_embeds") if "inputs_embeds" in kw else (a[3] if len(a) > 3 else None)
        if embeds is not None and embeds[1] is None:
            with rec.region("vlm_prefix"):
                out = _orig(*a, **kw)
            if rec.account and rec._active:
                pkv = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
                rec.kv_bytes["vlm_prefix_kv_cache"] = tensor_bytes(pkv)
            return out
        return _orig(*a, **kw)

    pg.forward = types.MethodType(forward, pg)
    undo.append(lambda: setattr(pg, "forward", orig_forward))

    orig_denoise = model.denoise_step

    def denoise_step(self, *a, _orig=orig_denoise, **kw):
        rec.step_idx += 1
        i = rec.step_idx
        with rec.region(f"ae_step[{i}]", alias=f"ae_step_{i}"):
            return _orig(*a, **kw)

    model.denoise_step = types.MethodType(denoise_step, model)
    undo.append(lambda: setattr(model, "denoise_step", orig_denoise))

    def _undo():
        for f in reversed(undo):
            f()
    return _undo


# ====================================================================== GR00T
def groot_parts(model):
    """(vision_owner, vision_attr, decoder_module, lm_head_module, family) for this GR00T version; raises if not found.

    vlm_prefix is the LLM's DECODER STACK in every version. N1.5 / N1.6's eagle.language_model is a Qwen3ForCausalLM,
    whose forward also runs the vocabulary lm_head (151,680 x 2048); N1.7's inner.language_model is the bare decoder
    and its lm_head sits on the outer Qwen3VLForConditionalGeneration. GR00T discards the logits in all three (it keeps
    hidden states), so the lm_head is its own region, backbone_lm_head, in all three (T004 review: the first version
    wrapped the CausalLM for N1.5 / N1.6, putting the lm_head inside vlm_prefix, and outside it for N1.7)."""
    bb = model.backbone
    eagle = getattr(bb, "eagle_model", None)                       # N1.5
    if eagle is None and hasattr(getattr(bb, "model", None), "extract_feature"):
        eagle = bb.model                                           # N1.6
    if eagle is not None:
        lm = eagle.language_model
        return eagle, "extract_feature", lm.model, lm.lm_head, "eagle"
    qwen = getattr(bb, "model", None)                              # N1.7: Qwen3VLForConditionalGeneration
    inner = getattr(qwen, "model", None)
    if inner is not None and hasattr(inner, "get_image_features") and hasattr(inner, "language_model"):
        return inner, "get_image_features", inner.language_model, qwen.lm_head, "qwen3vl"
    raise SystemExit(f"cannot find the vision / LLM split in {type(bb).__name__}: refusing to time a region that "
                     f"would silently read 0 ms (the Orin's first N1.5 run did)")


def instrument_groot(policy, rec: Recorder, inner: bool = True):
    """inner=False (T005: the action head is compiled): no ae_step / ae_dit hooks, because they would sit inside the
    compiled graph and break it; ae_total (wrapped from outside) is then the only expert region."""
    model = policy.model
    ah = model.action_head
    undo = []
    vis_owner, vis_attr, lm, lm_head, family = groot_parts(model)

    def wrap_call(obj, attr, name):
        orig = getattr(obj, attr)

        def w(*a, _orig=orig, **k):
            with rec.region(name):
                return _orig(*a, **k)
        setattr(obj, attr, w)
        undo.append(lambda: setattr(obj, attr, orig))

    wrap_call(model, "get_action", "model_get_action")
    wrap_call(model.backbone, "forward", "vlm_backbone")
    wrap_call(vis_owner, vis_attr, "vision")
    wrap_call(lm, "forward", "vlm_prefix")
    wrap_call(lm_head, "forward", "backbone_lm_head")

    orig_get_action = ah.get_action

    def get_action(*a, _orig=orig_get_action, **k):
        with rec.region("ae_total"):
            out = _orig(*a, **k)
            if rec.step_idx >= 0:
                rec.end(f"ae_step[{rec.step_idx}]")
        return out
    ah.get_action = get_action
    undo.append(lambda: setattr(ah, "get_action", orig_get_action))

    if not inner:
        def _undo0():
            for f in reversed(undo):
                f()
        return _undo0, family

    orig_enc = ah.action_encoder.forward

    def enc_forward(*a, _orig=orig_enc, **k):
        if rec.step_idx >= 0:
            rec.end(f"ae_step[{rec.step_idx}]")
        rec.step_idx += 1
        rec.begin(f"ae_step[{rec.step_idx}]", alias=f"ae_step_{rec.step_idx}")
        return _orig(*a, **k)
    ah.action_encoder.forward = enc_forward
    undo.append(lambda: setattr(ah.action_encoder, "forward", orig_enc))

    orig_dit = ah.model.forward

    def dit_forward(*a, _orig=orig_dit, **k):
        if rec.account and rec._active:
            enc = k.get("encoder_hidden_states")
            rec.kv_bytes[f"ae_dit_encoder_hidden_states"] = tensor_bytes(enc)
        with rec.region(f"ae_dit[{rec.step_idx}]", alias=f"ae_dit_{rec.step_idx}"):
            return _orig(*a, **k)
    ah.model.forward = dit_forward
    undo.append(lambda: setattr(ah.model, "forward", orig_dit))

    def _undo():
        for f in reversed(undo):
            f()
    return _undo, family
