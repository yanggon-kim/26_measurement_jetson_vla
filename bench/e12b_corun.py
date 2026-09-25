"""E12b M2 / M3 -- the compiled co-run: whole GPU, and both stages together in green partitions.

DIAGNOSIS THAT SHAPES THIS HARNESS (see e12b_diagnose.py for the evidence run)
T014's compiled pipeline replayed `pg.embed_image@reduce-overhead` (prefix stream) and
`model.denoise_step@reduce-overhead` (expert stream) out of ONE CUDA-graph-tree pool, a structure
torch documents as assuming one sequential call stream, and fed the expert a KV cache the next
prefix could overwrite. This harness removes both by construction:

  * the expert is ONE torch.cuda.CUDAGraph of the whole N-step loop (hoisted timesteps), captured
    by this harness in its OWN memory pool from `denoise_step` compiled mode="default" -- the same
    Inductor kernels as the rung, without cudagraph trees  [owngraph_expert];
  * each frame's KV cache and pad mask are copied, on the PREFIX stream at the end of prefix(k),
    into one of two expert-owned buffers (k % 2); the expert graph captured for that parity reads
    only that buffer; prefix(k) first waits for expert(k-2), the last reader of that buffer
    [dbuf_kv]. The copy is counted in the prefix stage.
  * `graphs_both` additionally captures the whole prefix as this harness's own graph (embed_image
    compiled mode="default", modeling_pi0.py:681's att_masks conversion memoised because a graph
    cannot contain a host sync) [owngraph_prefix, memo_681].

GREEN CONTEXTS (M3). M0 measured that a graph CAPTURED on the primary context runs on the whole
GPU when replayed on a green stream (0.99x at 4 SMs). So in a split, each stage's graph is
captured INSIDE its own green context, and only graphs_both can be partitioned: the rung prefix
replays a graph tree recorded on the primary context and would escape its partition.

Timing: CUDA events on each stage's own stream, the start recorded AFTER the cross-stream wait;
T_pipe = wall / frames of a free-running pipeline gated only by events (no per-frame barrier,
which would serialise the stages). Host issue time is recorded per frame. Every action tensor is
counted for non-finite values on the device and read after the loop.
"""
from __future__ import annotations

import contextlib
import copy
import time

import torch

import corun as CR
import e12b_common as C


# ---------------------------------------------------------------------------------------
def leaves(x, out=None):
    """Tensors of a nested cache structure, in a deterministic order."""
    if out is None:
        out = []
    if torch.is_tensor(x):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            leaves(y, out)
    elif isinstance(x, dict):
        for k in sorted(x):
            leaves(x[k], out)
    elif hasattr(x, "__dict__"):
        for k in sorted(vars(x)):
            leaves(vars(x)[k], out)
    return out


def copy_tree(dst, src):
    a, b = leaves(dst), leaves(src)
    if len(a) != len(b):
        raise RuntimeError(f"cache structure changed: {len(a)} vs {len(b)} tensors")
    for d, s in zip(a, b):
        if d.shape != s.shape:
            raise RuntimeError(f"cache tensor shape changed {tuple(d.shape)} vs {tuple(s.shape)}")
        d.copy_(s, non_blocking=True)


def memo_681(policy) -> str:
    """Memoise ONLY modeling_pi0.py:681 (att_masks -> tensor in embed_prefix), for graphs_both.
    A captured graph cannot contain a host sync; the list is a constant for a given prompt
    length, keyed on its contents exactly as corun.patch_pi0_nosync does."""
    import lerobot.policies.pi0.modeling_pi0 as M                     # noqa: PLC0415
    m = policy.model
    real = torch.tensor
    memo: dict = {}

    def cached(data, **kw):
        if type(data) is list and data and isinstance(data[0], (bool, int, float)):
            key = (tuple(data), kw.get("dtype"), str(kw.get("device")))
            if key not in memo:
                memo[key] = real(data, **kw)
            return memo[key]
        return real(data, **kw)

    orig = m.embed_prefix

    def inner(*a, **k):
        M.torch.tensor = cached
        try:
            return orig(*a, **k)
        finally:
            M.torch.tensor = real
    m.embed_prefix = inner
    return "memo_681: modeling_pi0.py:681 att_masks->tensor memoised on contents (prefix only)"


def recompile_default(policy, spec, which: str) -> str:
    """Replace a rung-compiled callable with torch.compile(mode='default') of the ORIGINAL method:
    identical Inductor kernels, no cudagraph trees, so this harness can capture it itself."""
    import opt_ladder as OL                                            # noqa: PLC0415
    if which == "denoise_step":
        m = policy.model
        m.denoise_step = torch.compile(type(m).denoise_step.__get__(m), mode="default")
        return "model.denoise_step@default"
    if which == "embed_image":
        pg = OL.owner_of(policy, "pg", spec)
        pg.embed_image = torch.compile(type(pg).embed_image.__get__(pg), mode="default")
        return "pg.embed_image@default"
    raise ValueError(which)


# ---------------------------------------------------------------------------------------
class Harness:
    """One process, two stages, their graphs and buffers; full GPU or one green split."""

    def __init__(self, policy, spec, frames, N=10, config="rung_prefix", split=None,
                 order="prefix_first", prio=(0, 0)):
        self.policy, self.spec, self.frames, self.N = policy, spec, frames, N
        self.config, self.split, self.order = config, split, order
        self.m = policy.model
        self.prio = prio
        if split is None:
            self.gp = self.ge = None
            # priority: lower number = higher priority; torch.cuda.Stream.priority_range() on this
            # device is (0, -3). Read back after creation -- E11f's first version asked for a
            # priority and never checked it was granted.
            self.sp = torch.cuda.Stream(priority=prio[0])
            self.se = torch.cuda.Stream(priority=prio[1])
            self.prio_granted = (self.sp.priority, self.se.priority)
        else:
            import green_ctx as G                                      # noqa: PLC0415
            if config != "graphs_both":
                raise SystemExit("a split needs graphs_both: the rung prefix replays a graph tree "
                                 "recorded on the primary context, which M0 shows escapes a "
                                 "green partition")
            self.gp, self.ge = G.split_pair(*split)
            self.sp, self.se = self.gp.stream, self.ge.stream
        self.granted = (self.gp.granted_sm, self.ge.granted_sm) if split else (16, 16)
        self.pool_e = torch.cuda.graph_pool_handle()
        self.pool_p = torch.cuda.graph_pool_handle()
        self._build()

    # context managers: which context is current while issuing each stage
    # switch(), never activate(): activate() drains the device on exit and would serialise
    # the two stages -- manufacturing the no-overlap result this harness exists to test.
    def _on_p(self):
        return self.gp.switch() if self.gp else torch.cuda.stream(self.sp)

    def _on_e(self):
        return self.ge.switch() if self.ge else torch.cuda.stream(self.se)

    def label(self) -> str:
        t = ["bf16_both", "hoist", "owngraph_expert", "dbuf_kv"]
        if self.config == "graphs_both":
            t += ["owngraph_prefix", "memo_681"]
        if self.split:
            t.append(f"green_{self.granted[0]}_{self.granted[1]}")
        if not self.split and self.prio[0]:
            t.append(f"prio_prefix_{self.prio_granted[0]}")
        if not self.split and self.prio[1]:
            t.append(f"prio_expert_{self.prio_granted[1]}")
        import os                                                      # noqa: PLC0415
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"):
            t.append(f"maxconn_{os.environ['CUDA_DEVICE_MAX_CONNECTIONS']}")
        t.append(self.order)
        return "+".join(t)

    # ---- construction ----------------------------------------------------------------
    def _prefix_eager(self, ctx):
        return CR.prefix_stage(ctx)

    def _build(self):
        fr0 = self.frames[0]
        with torch.no_grad(), self._on_p():
            pkv0, pad0 = self._prefix_eager(fr0["ctx"])
            torch.cuda.synchronize()
            self.kv = [copy.deepcopy(pkv0), copy.deepcopy(pkv0)]
            self.pad = [pad0.clone(), pad0.clone()]
        with torch.no_grad(), self._on_e():
            self.noise = fr0["noise"].clone()
            self.state = fr0["ctx"]["state"].clone()
            self.ts = CR.timestep_tensors(self.N, pad0.shape[0], pad0.device)
            self.eg, self.eout = [None, None], [None, None]
            for p in (0, 1):
                for _ in range(3):                         # warm-up / compile, on the stream
                    self._expert_body(p)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool_e, stream=torch.cuda.current_stream()):
                    self.eout[p] = self._expert_body(p)
                self.eg[p] = g
            torch.cuda.synchronize()
        self.pg_graph = None
        if self.config == "graphs_both":
            with torch.no_grad(), self._on_p():
                c = fr0["ctx"]
                self.pin = {k: c[k].clone() for k in ("images", "img_masks", "lang", "lang_mask")
                            if torch.is_tensor(c[k])}
                if isinstance(c["images"], (list, tuple)):
                    self.pin["images"] = [t.clone() for t in c["images"]]
                    self.pin["img_masks"] = [t.clone() for t in c["img_masks"]]
                self.pctx = dict(c)
                self.pctx.update(self.pin)
                for _ in range(3):
                    self._prefix_eager(self.pctx)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool_p, stream=torch.cuda.current_stream()):
                    self.pout = self._prefix_eager(self.pctx)
                self.pg_graph = g
                torch.cuda.synchronize()

    def _expert_body(self, p):
        x = self.noise
        dt = -1.0 / self.N
        for step in range(self.N):
            v = self.m.denoise_step(self.state, self.pad[p], self.kv[p], x, self.ts[step])
            x = x + dt * v
        return x

    def _load_prefix_inputs(self, ctx):
        for k, t in self.pin.items():
            src = ctx[k]
            if isinstance(t, list):
                for d, s in zip(t, src):
                    d.copy_(s, non_blocking=True)
            else:
                t.copy_(src, non_blocking=True)

    # ---- one frame's stages (issued; nothing waits on the host) ----------------------
    def issue_prefix(self, k, ev_e_done, ev_p_done, ev_p0, ev_p1, stale=False):
        fr = self.frames[k % len(self.frames)]
        with torch.no_grad(), self._on_p() as _:
            s = torch.cuda.current_stream()
            if k >= 2 and ev_e_done[k - 2] is not None:
                s.wait_event(ev_e_done[k - 2])          # last reader of buffer k % 2
            ev_p0[k].record(s)
            if self.pg_graph is not None:
                self._load_prefix_inputs(fr["ctx"])
                self.pg_graph.replay()
                pkv, pad = self.pout
            else:
                pkv, pad = self._prefix_eager(fr["ctx"])
            b = k % 2
            copy_tree(self.kv[b], pkv)
            self.pad[b].copy_(pad, non_blocking=True)
            ev_p1[k].record(s)
            ev_p_done[k].record(s)

    def issue_expert(self, k, ev_p_done, ev_e_done, ev_e0, ev_e1, fin, keep=None, stale=False):
        fr = self.frames[k % len(self.frames)]
        with torch.no_grad(), self._on_e():
            s = torch.cuda.current_stream()
            s.wait_event(ev_p_done[k])
            ev_e0[k].record(s)
            self.noise.copy_(fr["noise"], non_blocking=True)
            self.state.copy_(fr["ctx"]["state"], non_blocking=True)
            # stale control: read the OTHER buffer = a different frame's KV
            p = (k + 1) % 2 if stale else k % 2
            self.eg[p].replay()
            out = self.eout[p]
            fin.add(out)
            if keep is not None:
                keep[k] = out.clone()
            ev_e1[k].record(s)
            ev_e_done[k].record(s)

    # ---- the four conditions ---------------------------------------------------------
    def _events(self, F):
        mk = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(F)]     # noqa: E731
        return {n: mk() for n in ("pd", "ed", "p0", "p1", "e0", "e1")}

    def run(self, cond, F, fin, keep=None, stale=False):
        """cond in {prefix_alone, expert_alone, serial, pipeline}. Returns per-frame records."""
        ev = self._events(F)
        ev_e_done = [None] * F
        for i in range(F):
            ev_e_done[i] = ev["ed"][i]
        host = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if cond == "prefix_alone":
            for k in range(F):
                a = time.perf_counter()
                self.issue_prefix(k, [None] * F, ev["pd"], ev["p0"], ev["p1"])
                host.append((time.perf_counter() - a) * 1e3)
        elif cond == "expert_alone":
            # fill both buffers once, then the expert alone, frame after frame
            for k in range(2):
                self.issue_prefix(k, [None] * F, ev["pd"], ev["p0"], ev["p1"])
            torch.cuda.synchronize()
            for i in range(F):
                ev["pd"][i] = ev["pd"][i % 2]
            t0 = time.perf_counter()
            for k in range(F):
                a = time.perf_counter()
                self.issue_expert(k, ev["pd"], ev_e_done, ev["e0"],
                                  ev["e1"], fin, keep)
                host.append((time.perf_counter() - a) * 1e3)
        elif cond == "serial":
            for k in range(F):
                a = time.perf_counter()
                self.issue_prefix(k, ev_e_done, ev["pd"], ev["p0"], ev["p1"])
                self.issue_expert(k, ev["pd"], ev_e_done, ev["e0"], ev["e1"], fin, keep)
                host.append((time.perf_counter() - a) * 1e3)
                ev_e_done[k].synchronize()              # serial: frame k done before k+1
        elif cond == "pipeline":
            for k in range(F + 1):
                a = time.perf_counter()
                if self.order == "prefix_first":
                    if k < F:
                        self.issue_prefix(k, ev_e_done, ev["pd"], ev["p0"], ev["p1"])
                    if k >= 1:
                        self.issue_expert(k - 1, ev["pd"], ev_e_done, ev["e0"], ev["e1"], fin,
                                          keep, stale=stale)
                else:
                    if k >= 1:
                        self.issue_expert(k - 1, ev["pd"], ev_e_done, ev["e0"], ev["e1"], fin,
                                          keep, stale=stale)
                    if k < F:
                        self.issue_prefix(k, ev_e_done, ev["pd"], ev["p0"], ev["p1"])
                host.append((time.perf_counter() - a) * 1e3)
        else:
            raise ValueError(cond)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
        rec = []
        for k in range(F):
            r = {"frame": k}
            if cond in ("prefix_alone", "serial", "pipeline"):
                r["prefix_ms"] = ev["p0"][k].elapsed_time(ev["p1"][k])
            if cond in ("expert_alone", "serial", "pipeline"):
                r["expert_ms"] = ev["e0"][k].elapsed_time(ev["e1"][k])
            if cond == "serial":
                r["frame_ms"] = ev["p0"][k].elapsed_time(ev["e1"][k])
            rec.append(r)
        return {"records": rec, "wall_ms": wall, "host_issue_ms": host,
                "period_ms": wall / F}


@contextlib.contextmanager
def nullctx():
    yield
