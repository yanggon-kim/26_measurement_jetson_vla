"""Protocol §2 region instrumentation for LeRobot's π0, by wrapping — never by editing.

Regions: `preprocess`, `vision`, `vlm_prefix`, `ae_total`, `ae_step[i]`, `ae_layer[j]`,
`postprocess`. Every region emits an NVTX range (for T004's nsys/ncu) and a pair of CUDA
events (for protocol §2 timing). Events are only read after an explicit synchronise at the
end of the iteration, so nothing is serialised mid-inference.

Wrap points, all found in the installed `lerobot/policies/pi0/modeling_pi0.py`:

  vision      PaliGemmaWithExpertModel.embed_image      — once per camera (3 here)
  vlm_prefix  PaliGemmaWithExpertModel.forward          — the `inputs_embeds[1] is None` branch
  ae_step[i]  PI0Pytorch.denoise_step                   — once per denoising step
  ae_layer[j] gemma_expert.model.layers[j].forward      — per-instance; the class is built by
                                                          a factory and has no importable name

`ae_total` is not wrapped: it is derived as the span from the first `ae_step` start event to
the last `ae_step` end event, which is exactly the denoising loop and costs no extra events.

Each range is emitted twice, once with brackets (`ae_step[0]`) and once without
(`ae_step_0`), because protocol §3.2 warns that ncu's `--nvtx-include` may not parse brackets.
"""
from __future__ import annotations

import types
from contextlib import contextmanager

import torch


class Recorder:
    """Collects (region, start_event, end_event) triples for one iteration."""

    def __init__(self, enable_layers: bool = False, layers_in_step: int = 0):
        self.enable_layers = enable_layers
        self.layers_in_step = layers_in_step
        self.spans: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.step_idx = -1
        self._active = False

    def reset(self) -> None:
        self.spans.clear()
        self.step_idx = -1

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    @contextmanager
    def region(self, name: str, alias: str | None = None):
        if not self._active:
            yield
            return
        torch.cuda.nvtx.range_push(name)
        if alias:                       # bracket-free duplicate for ncu --nvtx-include
            torch.cuda.nvtx.range_push(alias)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            yield
        finally:
            e.record()
            self.spans.append((name, s, e))
            if alias:
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

    # ---------------------------------------------------------------- results

    def results_ms(self) -> dict[str, float]:
        """Call only after torch.cuda.synchronize(). Sums repeats of the same name."""
        out: dict[str, float] = {}
        for name, s, e in self.spans:
            out[name] = out.get(name, 0.0) + s.elapsed_time(e)
        # ae_total = first ae_step start -> last ae_step end (the denoising loop)
        steps = [(n, s, e) for n, s, e in self.spans if n.startswith("ae_step[")]
        if steps:
            out["ae_total"] = steps[0][1].elapsed_time(steps[-1][2])
        return out

    def step_times_ms(self) -> list[float]:
        return [s.elapsed_time(e) for n, s, e in self.spans if n.startswith("ae_step[")]


def instrument(policy, rec: Recorder):
    """Monkeypatch the wrap points onto this policy instance. Returns an undo callable."""
    import lerobot.policies.pi0.modeling_pi0 as M

    model = policy.model
    pg = model.paligemma_with_expert
    undo = []

    # ---- vision: once per camera
    orig_embed_image = pg.embed_image

    def embed_image(self, image, _orig=orig_embed_image):
        with rec.region("vision"):
            return _orig(image)

    pg.embed_image = types.MethodType(embed_image, pg)
    undo.append(lambda: setattr(pg, "embed_image", orig_embed_image))

    # ---- vlm_prefix: the prefill branch of the fused forward
    orig_forward = pg.forward

    def forward(self, *a, _orig=orig_forward, **kw):
        embeds = kw.get("inputs_embeds") or (a[3] if len(a) > 3 else None)
        is_prefill = embeds is not None and embeds[1] is None
        if is_prefill:
            with rec.region("vlm_prefix"):
                return _orig(*a, **kw)
        return _orig(*a, **kw)

    pg.forward = types.MethodType(forward, pg)
    undo.append(lambda: setattr(pg, "forward", orig_forward))

    # ---- ae_step[i]
    orig_denoise = model.denoise_step

    def denoise_step(self, *a, _orig=orig_denoise, **kw):
        rec.step_idx += 1
        i = rec.step_idx
        with rec.region(f"ae_step[{i}]", alias=f"ae_step_{i}"):
            return _orig(*a, **kw)

    model.denoise_step = types.MethodType(denoise_step, model)
    undo.append(lambda: setattr(model, "denoise_step", orig_denoise))

    # ---- ae_layer[j], only inside one representative step (protocol §2)
    if rec.enable_layers:
        layers = pg.gemma_expert.model.layers
        for j, layer in enumerate(layers):
            orig_lf = layer.forward

            def lf(*a, _orig=orig_lf, _j=j, **kw):
                if rec.step_idx == rec.layers_in_step:
                    with rec.region(f"ae_layer[{_j}]", alias=f"ae_layer_{_j}"):
                        return _orig(*a, **kw)
                return _orig(*a, **kw)

            layer.forward = lf
            undo.append(lambda l=layer, o=orig_lf: setattr(l, "forward", o))

    def _undo():
        for f in reversed(undo):
            f()

    return _undo


@contextmanager
def nvtx_only(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
