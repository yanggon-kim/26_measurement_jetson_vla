"""T005 / E2 — the action expert one kernel at a time.

E3/E4 measured the action expert as a block: 52.8 ms per denoising step eager, 17x above its
analytic floor. E2 asks where that factor of 17 lives, by taking the expert's own GEMMs out of
the model and running them alone across sequence lengths.

The shapes are read from the loaded checkpoint, never from a table. They are:

    hidden 1024, intermediate 4096, 18 layers, 8 query heads, **1 KV head**, head_dim 256

    q_proj     1024 -> 2048        gate_proj  1024 -> 4096
    k_proj     1024 ->  256        up_proj    1024 -> 4096
    v_proj     1024 ->  256        down_proj  4096 -> 1024
    o_proj     2048 -> 1024

`docs/01_measurement_plan.md` lists E2's `kv_proj (k and v)` as one 1024 -> 512 projection.
The expert is multi-query, so k and v are two *separate* 1024 -> 256 GEMMs. Same weight bytes,
but two kernels rather than one — which is the whole question at S = 51, where launch cost and
weight streaming dominate and FLOPs do not.

What it produces:

  * `kernels.csv` — per projection per S: time, FLOPs, bytes, arithmetic intensity, the
    memory and compute floors from E1's *measured* peaks, and the multiple of floor achieved;
  * `launch_cost.csv` — the cost of a launch, measured four ways, and what it implies for a
    step made of 18 x 7 = 126 projection launches;
  * `cuda_graph.csv` — the same 18-layer stack replayed as a graph, which separates "the
    kernels are slow" from "there are too many launches".

    python bench/micro_ae_kernels.py --outdir results/<date>_E2_ae_kernels
    python bench/micro_ae_kernels.py --outdir ... --ncu-window q_proj@51        # T007's ncu target
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
from common import (REPO, collect_raw, nvtx, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

DEFAULT_S = [1, 2, 4, 8, 16, 32, 51, 64, 128, 256]
ALL_SAMPLES: list[dict] = []          # every timed call, dumped to raw/ (protocol §3.5)


def stash(proj, S, mode, samples):
    for i, v in enumerate(samples):
        ALL_SAMPLES.append({"proj": proj, "S": S, "mode": mode, "iter": i, "ms": v})
    return samples
BYTES_PER_EL = 2          # bf16 weights and activations


def peaks() -> dict:
    """E1's measured peaks. Refuse to fall back to spec values (protocol §8)."""
    f = REPO / "results" / "20260921_E1_peak" / "hw_constants.json"
    if not f.exists():
        raise SystemExit("E1 has not run; its measured peaks are required. "
                         "Run bench/micro_peak.py first.")
    mp = json.loads(f.read_text())["measured_peaks"]
    if not mp.get("peak_memory_bandwidth_GBps") or not mp.get("peak_bf16_TFLOPps"):
        raise SystemExit("E1's hw_constants.json has no measured peaks.")
    return mp


def expert_geometry(policy):
    """Read the real shapes out of the checkpoint."""
    ex = policy.model.paligemma_with_expert.gemma_expert
    c = ex.config
    layers = ex.model.layers
    l0 = layers[0]
    order = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    mods = dict(l0.named_modules())
    projs = []
    for name in order:
        m = mods[name]
        projs.append({"name": name.split(".")[-1], "module": m,
                      "in": m.in_features, "out": m.out_features})
    cfg = {"hidden_size": c.hidden_size, "intermediate_size": c.intermediate_size,
           "num_hidden_layers": c.num_hidden_layers,
           "num_attention_heads": c.num_attention_heads,
           "num_key_value_heads": c.num_key_value_heads, "head_dim": c.head_dim}
    return projs, layers, cfg


def timed(fn, warmup: int = 20, n: int = 100) -> list[float]:
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


def chain_marginal(fn, ks=(1, 8, 64)) -> tuple[float, float, float]:
    """Marginal cost of ONE call, the fixed cost of a timed region, and the fit residual.

    A single call measured between one CUDA-event pair is not the cost of that call: on this
    device the event pair plus loop costs a few tens of microseconds, comparable to the call
    itself. An earlier version of this experiment multiplied that constant by 18 and 36 and
    reported it as model time. Timing chains of k calls between ONE event pair and fitting
    t(k) = fixed + k * marginal separates the two.

    Three points, least squares, and the residual is reported per row -- a two-point fit gives
    no error bar, and the fixed term turns out to be regime-dependent rather than constant
    (it collapses towards zero once the call is GPU-bound, because then the chain is limited by
    the GPU and not by the launch path).

    `marginal` is the cost of one call issued back to back, which is what a model does. Because
    CUDA launches are asynchronous it is max(CPU dispatch, GPU time), NOT CPU time alone.

    Returns (marginal_ms, fixed_ms, max_abs_residual_ms).
    """
    ys = []
    for k in ks:
        def chain(k=k):
            for _ in range(k):
                fn()
        ys.append(summarize(timed(chain, 20, 100))["median_ms"])
    n = len(ks)
    mx = sum(ks) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in ks)
    marginal = sum((x - mx) * (y - my) for x, y in zip(ks, ys)) / sxx
    fixed = my - marginal * mx
    resid = max(abs(y - (fixed + marginal * x)) for x, y in zip(ks, ys))
    return marginal, fixed, resid


def capture(run):
    """Capture `run` as a CUDA graph. Replaying it removes the Python interpreter and the
    CUDA launch path from the timed region, leaving GPU kernel time."""
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(st)
    with torch.cuda.graph(g):
        run()
    return g


def kernel_rows(projs, layers, cfg, S_list, mp, use_nvtx: bool, only: str | None) -> list[dict]:
    """Per projection, per sequence length, three times over.

    A first attempt timed one `F.linear` call per CUDA-event pair. Every projection came out
    at 0.10 ms regardless of shape, and an empty kernel at 68 us: at these sizes the
    interpreter cannot feed the GPU, so that measured Python dispatch, not the kernel. The
    numbers here are therefore taken three ways:

      `eager_single_ms`  one Python-driven call between one CUDA-event pair. This carries a
                     FIXED ~45 us cost for the timed region itself, comparable to the call,
                     so it is reported but never multiplied out into a per-step budget;
  `eager_chain_ms`   the marginal cost of one call inside a 64-call chain, from fitting
                     t(k) = fixed + k * marginal. THIS is what an operation costs when the
                     model issues it back to back;
      `graph_cold_ms` the same projection across all 18 layers inside one CUDA graph, so the
                      weights (18 x the matrix, far past the 4 MB L2) stream from DRAM as they
                      do in the real model. This is the honest kernel time;
      `graph_hot_ms`  layer 0's weights replayed 18 times, so they sit in L2. The gap between
                      hot and cold is what a memory system that kept the weights close would
                      be worth -- which is the question the PIM study is asking.

    Floors and `x_floor` are computed against `graph_cold_ms`.
    """
    bw = mp["peak_memory_bandwidth_GBps"] * 1e9
    fl = mp["peak_bf16_TFLOPps"] * 1e12
    nl = cfg["num_hidden_layers"]
    full = [dict(l.named_modules()) for l in layers]
    rows = []
    for p in projs:
        if only and p["name"] != only:
            continue
        path = {"q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
                "v_proj": "self_attn.v_proj", "o_proj": "self_attn.o_proj",
                "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
                "down_proj": "mlp.down_proj"}[p["name"]]
        ws = [full[i][path].weight for i in range(nl)]
        w0 = ws[0]
        for S in S_list:
            x = torch.randn(S, p["in"], device="cuda", dtype=w0.dtype)
            F = torch.nn.functional.linear

            def one(x=x, w0=w0):
                return F(x, w0)

            fn = one
            if use_nvtx:
                tag = f"e2/{p['name']}/S{S}"

                def fn(one=one, tag=tag):             # noqa: E731
                    with nvtx(tag):
                        return one()

            eager = summarize(stash(p["name"], S, "eager_single", timed(fn, 20, 100)))
            marg, fixed, resid = chain_marginal(fn)

            def cold(x=x, ws=ws, F=F):
                for w in ws:
                    F(x, w)

            def hot(x=x, w0=w0, nl=nl, F=F):
                for _ in range(nl):
                    F(x, w0)

            gc_, gh_ = None, None
            try:
                g = capture(cold)
                gc_ = summarize(stash(p["name"], S, "graph_cold_18x",
                                      timed(lambda g=g: g.replay(), 20, 100)))
                del g
                torch.cuda.empty_cache()
                g = capture(hot)
                gh_ = summarize(stash(p["name"], S, "graph_hot_18x",
                                      timed(lambda g=g: g.replay(), 20, 100)))
                del g
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    graph capture failed for {p['name']} S={S}: "
                      f"{type(e).__name__}: {str(e)[:80]}")

            flops = 2 * S * p["in"] * p["out"]
            wb = p["in"] * p["out"] * BYTES_PER_EL
            ab = S * (p["in"] + p["out"]) * BYTES_PER_EL
            nb = wb + ab
            mem_floor = nb / bw * 1e3
            cmp_floor = flops / fl * 1e3
            floor = max(mem_floor, cmp_floor)
            cold_ms = gc_["median_ms"] / nl if gc_ else None
            hot_ms = gh_["median_ms"] / nl if gh_ else None
            sec = (cold_ms / 1e3) if cold_ms else None
            rows.append({
                "proj": p["name"], "S": S, "in": p["in"], "out": p["out"],
                "eager_single_ms": round(eager["median_ms"], 6),
                "eager_single_mean_ms": round(eager["mean_ms"], 6),
                "eager_single_p5_ms": round(eager["p5_ms"], 6),
                "eager_single_p95_ms": round(eager["p95_ms"], 6),
                "eager_single_min_ms": round(eager["min_ms"], 6),
                "eager_chain_ms": round(marg, 6),
                "timed_region_fixed_ms": round(fixed, 6),
                "chain_fit_max_resid_ms": round(resid, 6),
                "graph_cold_ms": round(cold_ms, 6) if cold_ms else "",
                "graph_cold_mean_ms": round(gc_["mean_ms"] / nl, 6) if gc_ else "",
                "graph_cold_p5_ms": round(gc_["p5_ms"] / nl, 6) if gc_ else "",
                "graph_cold_p95_ms": round(gc_["p95_ms"] / nl, 6) if gc_ else "",
                "graph_cold_min_ms": round(gc_["min_ms"] / nl, 6) if gc_ else "",
                "graph_hot_ms": round(hot_ms, 6) if hot_ms else "",
                "graph_hot_p5_ms": round(gh_["p5_ms"] / nl, 6) if gh_ else "",
                "graph_hot_p95_ms": round(gh_["p95_ms"] / nl, 6) if gh_ else "",
                # The chained marginal cost. Launches are asynchronous, so a chain runs at
                # max(CPU dispatch, GPU time) per call -- this column is that maximum, NOT
                # CPU time alone, and NOT (eager - graph), which would understate the CPU
                # cost by assuming the two serialise. Where it sits at ~62 us while GPU time
                # is well below that, the call is CPU-bound and 62 us IS the dispatch cost;
                # where it tracks gpu_time_us the call has become GPU-bound.
                "marginal_us": round(marg * 1e3, 2),
                "gpu_time_us": round(cold_ms * 1e3, 2) if cold_ms else "",
                "marginal_over_gpu": round(marg / cold_ms, 2) if cold_ms else "",
                "regime": ("cpu_bound (dispatch)" if cold_ms and marg > 1.15 * cold_ms
                           else "gpu_bound") if cold_ms else "",
                "hot_speedup": round(cold_ms / hot_ms, 3) if cold_ms and hot_ms else "",
                "n_eager": eager["n"], "n_graph": gc_["n"] if gc_ else 0,
                "flops": flops, "weight_bytes": wb, "act_bytes": ab, "total_bytes": nb,
                "floor_bytes_definition": "weights + input activations + output activations",
                "graph_cold_working_set_bytes": wb * nl,
                "graph_cold_exceeds_L2": wb * nl > 4 * 1024 * 1024,
                "weight_frac_of_bytes": round(wb / nb, 4),
                "arith_intensity_FLOP_per_B": round(flops / nb, 3),
                "ridge_FLOP_per_B": mp["ridge_point_FLOP_per_byte"],
                "bound_by": "memory" if flops / nb < mp["ridge_point_FLOP_per_byte"]
                            else "compute",
                "mem_floor_ms": round(mem_floor, 6), "compute_floor_ms": round(cmp_floor, 6),
                "floor_ms": round(floor, 6),
                "x_floor": round(cold_ms / floor, 2) if cold_ms else "",
                "achieved_GBps": round(nb / sec / 1e9, 2) if sec else "",
                "achieved_TFLOPps": round(flops / sec / 1e12, 4) if sec else "",
                "pct_of_peak_bw": round(100 * (nb / sec) / bw, 1) if sec else "",
                "pct_of_peak_flops": round(100 * (flops / sec) / fl, 1) if sec else "",
            })
            del x
            torch.cuda.empty_cache()
    return rows


def op_rows(cfg, S_list, mp, P_tokens: int = 816) -> list[dict]:
    """The expert's non-GEMM operators, which is where a denoising step actually spends itself.

    E3 measured 52.8 ms per eager denoising step. The 126 projection GEMMs of that step account
    for 8.6 ms of it. So the interesting question for E2 is not the GEMMs -- it is attention
    against the 816-token prefix KV cache, and the pointwise work around it.

    Attention here is S queries (8 heads x 256) against T = P + S keys with **one** KV head,
    which is the real shape: the expert is multi-query.
    """
    bw = mp["peak_memory_bandwidth_GBps"] * 1e9
    H = cfg["num_attention_heads"]
    D = cfg["head_dim"]
    Hn = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    dt = torch.bfloat16
    rows = []
    for S in S_list:
        T = P_tokens + S
        q = torch.randn(1, H, S, D, device="cuda", dtype=dt)
        k = torch.randn(1, 1, T, D, device="cuda", dtype=dt)
        v = torch.randn(1, 1, T, D, device="cuda", dtype=dt)
        ke = k.expand(1, H, T, D)
        ve = v.expand(1, H, T, D)
        h = torch.randn(S, Hn, device="cuda", dtype=dt)
        h2 = torch.randn(S, Hn, device="cuda", dtype=dt)
        mlp = torch.randn(S, I, device="cuda", dtype=dt)
        scores = torch.randn(1, H, S, T, device="cuda", dtype=dt)
        rope = torch.randn(1, H, S, D, device="cuda", dtype=dt)
        cos = torch.randn(1, 1, S, D, device="cuda", dtype=dt)
        sin = torch.randn(1, 1, S, D, device="cuda", dtype=dt)
        norm_w = torch.randn(Hn, device="cuda", dtype=dt)

        def rms(h=h, w=norm_w):
            x = h.float()
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dt) * w

        def rope_fn(rope=rope, cos=cos, sin=sin):
            # the standard rotate-half form: out = x*cos + rotate_half(x)*sin
            x1, x2 = rope[..., : D // 2], rope[..., D // 2:]
            return rope * cos + torch.cat((-x2, x1), dim=-1) * sin

        ops = [
            ("sdpa_attention", lambda q=q, ke=ke, ve=ve:
                torch.nn.functional.scaled_dot_product_attention(q, ke, ve),
             # q read + K,V read + out. K and V are counted ONCE each: there is one KV head
             # and `expand` is a view, so the unique bytes touched are T*D per tensor however
             # many query heads read them. Unique bytes is the right quantity for a floor.
             S * H * D * 2 + 2 * T * D * 2 + S * H * D * 2,
             2 * 2 * H * S * T * D,
             f"S={S} queries (8 heads x 256) against T={T} keys, 1 KV head"),
            ("softmax", lambda scores=scores: torch.softmax(scores.float(), dim=-1),
             H * S * T * 2 + H * S * T * 4, 0, f"over [1,{H},{S},{T}]"),
            ("rmsnorm", rms, S * Hn * 2 * 2 + Hn * 2, 0, f"[{S},{Hn}]"),
            ("rope", rope_fn, S * H * D * 2 * 2, 0, f"[1,{H},{S},{D}]"),
            ("gelu", lambda mlp=mlp: torch.nn.functional.gelu(mlp, approximate="tanh"),
             S * I * 2 * 2, 0, f"[{S},{I}]"),
            ("elementwise_add", lambda h=h, h2=h2: h + h2, S * Hn * 2 * 3, 0, f"[{S},{Hn}]"),
        ]
        for name, fn, nb, flops, note in ops:
            eager = summarize(stash(name, S, "eager_single", timed(fn, 20, 100)))
            marg, fixed, resid = chain_marginal(fn)
            gms = None
            try:
                def rep(fn=fn):
                    for _ in range(18):
                        fn()
                g = capture(rep)
                gr = summarize(stash(name, S, "graph_18x",
                                     timed(lambda g=g: g.replay(), 20, 100)))
                gms = gr["median_ms"] / 18
                del g
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    graph capture failed for {name} S={S}: {type(e).__name__}")
            sec = (gms / 1e3) if gms else None
            # These operators run on activations, not weights: at the model's own sizes the
            # whole working set is a couple of MB and never leaves the 4 MB L2. So `bytes`
            # here is L2 traffic, NOT off-chip traffic, and dividing it by a DRAM peak is
            # meaningless -- several rows exceed 100 % of "peak" for exactly that reason,
            # which is the proof rather than an anomaly. The columns are named accordingly
            # and a DRAM-roofline column is deliberately NOT provided.
            ws = nb
            L2 = 4 * 1024 * 1024
            l2 = ws < L2            # strict: a working set of exactly the L2 size is not
                                    # safely resident, and `gelu` at S=256 hits it exactly
            rows.append({
                "op": name, "S": S, "T": T,
                "eager_single_ms": round(eager["median_ms"], 6),
                "eager_single_mean_ms": round(eager["mean_ms"], 6),
                "eager_single_p5_ms": round(eager["p5_ms"], 6),
                "eager_single_p95_ms": round(eager["p95_ms"], 6),
                "eager_chain_ms": round(marg, 6),
                "timed_region_fixed_ms": round(fixed, 6),
                "chain_fit_max_resid_ms": round(resid, 6),
                "marginal_us": round(marg * 1e3, 2),
                "chain_fit_max_resid_ms": round(resid, 6),
                "graph_ms": round(gms, 6) if gms else "",
                "graph_p5_ms": round(gr["p5_ms"] / 18, 6) if gms else "",
                "graph_p95_ms": round(gr["p95_ms"] / 18, 6) if gms else "",
                "gpu_time_us": round(gms * 1e3, 2) if gms else "",
                "marginal_over_gpu": round(marg / gms, 2) if gms else "",
                "regime": ("cpu_bound (dispatch)" if gms and marg > 1.15 * gms
                           else "gpu_bound") if gms else "",
                "n": eager["n"],
                "bytes_touched": nb, "flops": flops,
                "working_set_bytes": ws, "l2_resident": l2,
                "working_set_over_l2": round(ws / L2, 3),
                "l2_GBps": round(nb / sec / 1e9, 2) if sec else "",
                "dram_floor_ms_NOT_APPLICABLE_IF_L2_RESIDENT": round(nb / bw * 1e3, 6),
                "note": note + ("; working set fits the 4 MB L2, so l2_GBps is CACHE "
                                "bandwidth and must not be compared with the DRAM peak"
                                if l2 else "; working set exceeds the 4 MB L2"),
            })
        del q, k, v, ke, ve, h, h2, mlp, scores, rope, cos, sin, norm_w
        torch.cuda.empty_cache()
    return rows


def launch_rows(projs, mp) -> list[dict]:
    """How much of a projection's time is the launch itself.

    Three independent estimates, because no single one is trustworthy: an empty kernel, the
    smallest possible real GEMM (S=1), and the difference between a chain of n back-to-back
    launches and n times one launch.
    """
    rows = []
    e = torch.empty(1, device="cuda")
    ms = timed(lambda: e.add_(1.0), 50, 500)
    s = summarize(ms)
    rows.append({"method": "empty_kernel", "detail": "add_ on a 1-element tensor",
                 "median_ms": round(s["median_ms"], 6), "us": round(s["median_ms"] * 1e3, 2),
                 "n": s["n"]})

    q = next(p for p in projs if p["name"] == "q_proj")
    x1 = torch.randn(1, q["in"], device="cuda", dtype=q["module"].weight.dtype)
    ms = timed(lambda: torch.nn.functional.linear(x1, q["module"].weight), 50, 500)
    s = summarize(ms)
    rows.append({"method": "smallest_real_gemm", "detail": "q_proj at S=1 (1x1024 @ 1024x2048)",
                 "median_ms": round(s["median_ms"], 6), "us": round(s["median_ms"] * 1e3, 2),
                 "n": s["n"]})

    # a chain of k launches, to separate per-launch cost from per-kernel work
    x = torch.randn(51, q["in"], device="cuda", dtype=q["module"].weight.dtype)
    w = q["module"].weight
    for k in (1, 8, 64):
        ms = timed(lambda k=k: [torch.nn.functional.linear(x, w) for _ in range(k)], 20, 100)
        s = summarize(ms)
        rows.append({"method": f"chain_{k}", "detail": f"{k} back-to-back q_proj at S=51, "
                                                       f"Python-driven",
                     "median_ms": round(s["median_ms"], 6),
                     "us": round(s["median_ms"] * 1e3 / k, 2), "n": s["n"]})

    # the same chain inside a CUDA graph: no interpreter, no launch path. The difference
    # against chain_64 is the dispatch cost per operation.
    def chain64():
        for _ in range(64):
            torch.nn.functional.linear(x, w)

    try:
        g = capture(chain64)
        s = summarize(timed(lambda: g.replay(), 20, 100))
        rows.append({"method": "chain_64_cuda_graph",
                     "detail": "same 64 q_proj calls replayed as one graph (GPU time only)",
                     "median_ms": round(s["median_ms"], 6),
                     "us": round(s["median_ms"] * 1e3 / 64, 2), "n": s["n"]})
        del g
        torch.cuda.empty_cache()
    except Exception as e:
        rows.append({"method": "chain_64_cuda_graph", "detail": f"FAILED: {type(e).__name__}",
                     "median_ms": "", "us": "", "n": 0})
    return rows


def graph_rows(projs, layers, cfg, mp, S: int = 51) -> list[dict]:
    """The whole 18-layer projection stack, eager vs replayed as one CUDA graph.

    If the graph is much faster, the expert is launch-bound and the fix is CUDA graphs, not
    a faster memory system — which is exactly what the PIM argument needs to be tested against.
    """
    nl = cfg["num_hidden_layers"]
    h = cfg["hidden_size"]
    x = torch.randn(S, h, device="cuda", dtype=projs[0]["module"].weight.dtype)
    stack = []
    for li in range(nl):
        mods = dict(layers[li].named_modules())
        stack.append([mods[n] for n in ("self_attn.q_proj", "self_attn.k_proj",
                                        "self_attn.v_proj", "self_attn.o_proj",
                                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")])

    def run_stack():
        """Projections only, in the real order and at the real widths. Not the real maths —
        attention and the activation are left out on purpose, so this measures exactly the
        126 GEMM launches and nothing else."""
        for q, k, v, o, g, u, d in stack:
            a = torch.nn.functional.linear(x, q.weight)
            torch.nn.functional.linear(x, k.weight)
            torch.nn.functional.linear(x, v.weight)
            torch.nn.functional.linear(a, o.weight)
            gg = torch.nn.functional.linear(x, g.weight)
            torch.nn.functional.linear(x, u.weight)
            torch.nn.functional.linear(gg, d.weight)

    rows = []
    ms = timed(run_stack, 20, 100)
    s = summarize(ms)
    n_launch = nl * 7
    wb_layer = sum(m.weight.numel() * BYTES_PER_EL for m in stack[0])
    wb_total = wb_layer * nl
    rows.append({"mode": "eager", "S": S, "n_launches": n_launch,
                 "median_ms": round(s["median_ms"], 6),
                 "mean_ms": round(s["mean_ms"], 6),
                 "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                 "min_ms": round(s["min_ms"], 6), "n": s["n"],
                 "us_per_launch": round(s["median_ms"] * 1e3 / n_launch, 2),
                 "weight_bytes_total": wb_total,
                 "act_bytes_total": sum(S * (m.in_features + m.out_features) * BYTES_PER_EL
                                        for m in stack[0]) * nl,
                 "floor_bytes_definition": "WEIGHTS ONLY (see weight_bytes_total); the "
                                           "per-projection floors in kernels.csv add "
                                           "activations, so the two are not the same floor",
                 "note": "issued back to back from Python"})

    try:
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                run_stack()
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g):
            run_stack()
        ms = timed(lambda: g.replay(), 20, 100)
        s = summarize(ms)
        rows.append({"mode": "cuda_graph_replay", "S": S, "n_launches": n_launch,
                     "median_ms": round(s["median_ms"], 6),
                     "mean_ms": round(s["mean_ms"], 6),
                     "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                     "min_ms": round(s["min_ms"], 6), "n": s["n"],
                     "us_per_launch": round(s["median_ms"] * 1e3 / n_launch, 2),
                     "weight_bytes_total": wb_total,
                     "act_bytes_total": "", "floor_bytes_definition": "WEIGHTS ONLY",
                     "note": "one graph launch replaces 126 kernel launches"})
    except Exception as e:
        rows.append({"mode": "cuda_graph_replay", "S": S, "n_launches": n_launch,
                     "median_ms": "", "mean_ms": "", "p5_ms": "", "p95_ms": "", "min_ms": "",
                     "n": 0, "us_per_launch": "", "weight_bytes_total": wb_total,
                     "act_bytes_total": "", "floor_bytes_definition": "WEIGHTS ONLY",
                     "note": f"FAILED: {type(e).__name__}: {str(e)[:120]}"})
    return rows


def ncu_window(projs, spec: str) -> int:
    """One kernel, once, inside NVTX range `micro`, with the profiler started only for it.

    T007 profiles a single action-expert GEMM with Nsight Compute. Handing ncu the whole
    script would replay thousands of launches; this gives it exactly one.
    """
    name, _, s_str = spec.partition("@")
    S = int(s_str or 51)
    p = next((p for p in projs if p["name"] == name), None)
    if p is None:
        raise SystemExit(f"unknown projection {name!r}; have "
                         f"{[x['name'] for x in projs]}")
    w = p["module"].weight
    x = torch.randn(S, p["in"], device="cuda", dtype=w.dtype)
    for _ in range(50):                       # warm up outside the profiled window
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    with nvtx("micro"):
        torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    print(f"profiled one {name} at S={S}: {S}x{p['in']} @ {p['in']}x{p['out']}, "
          f"weights {p['in']*p['out']*BYTES_PER_EL/1e6:.2f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--S", default=",".join(map(str, DEFAULT_S)),
                    help="comma-separated sequence lengths")
    ap.add_argument("--layer", "--only", dest="layer", default=None,
                    help="restrict to one projection, e.g. q_proj")
    ap.add_argument("--nvtx", action="store_true",
                    help="wrap each measured kernel in an NVTX range")
    ap.add_argument("--ncu-window", dest="ncu_window", default=None,
                    help="T007's ncu target: `<proj>@<S>`, e.g. q_proj@51. Warms the kernel "
                         "up, then runs it ONCE between cudaProfilerStart/Stop inside an "
                         "NVTX range named `micro`, and exits. Profile it with "
                         "`sudo ncu --profile-from-start off --nvtx --nvtx-include micro ...`")
    ap.add_argument("--skip-graph", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    S_list = [int(s) for s in args.S.split(",") if s.strip()]

    gate = P.gate()
    mp = peaks()
    print("gate:", json.dumps(gate))
    print(f"E1 peaks: {mp['peak_memory_bandwidth_GBps']} GB/s, "
          f"{mp['peak_bf16_TFLOPps']} TFLOP/s, ridge {mp['ridge_point_FLOP_per_byte']} FLOP/B")

    policy, cfg_obj, shipped = P.load_policy(num_inference_steps=4)
    real = P.assert_real_weights(policy)
    print("weights:", json.dumps(real))
    projs, layers, cfg = expert_geometry(policy)

    if args.ncu_window:
        return ncu_window(projs, args.ncu_window)

    print("expert geometry:", json.dumps(cfg))
    for p in projs:
        print(f"  {p['name']:10s} {p['in']:5d} -> {p['out']:5d}")

    print("\n=== per-projection sweep ===", flush=True)
    krows = kernel_rows(projs, layers, cfg, S_list, mp, args.nvtx, args.layer)
    print(f"{'proj':10s} {'S':>4s} {'1call_ms':>9s} {'chain_ms':>9s} {'gpu_ms':>8s} "
          f"{'cpu_us':>7s} {'hot_x':>6s} {'AI':>7s} {'floor_ms':>9s} {'xfloor':>7s} "
          f"{'%peakBW':>8s}")
    for r in krows:
        print(f"{r['proj']:10s} {r['S']:4d} {r['eager_single_ms']:9.4f} "
              f"{r['eager_chain_ms']:9.4f} {r['graph_cold_ms']:8.4f} "
              f"{r['marginal_us']:7.1f} {r['hot_speedup']:6.2f} "
              f"{r['arith_intensity_FLOP_per_B']:7.1f} "
              f"{r['floor_ms']:9.4f} {r['x_floor']:7.2f} {r['pct_of_peak_bw']:8.1f}")
    with (out / "kernels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(krows[0].keys()))
        w.writeheader()
        w.writerows(krows)

    print("\n=== non-GEMM operators (attention, norms, pointwise) ===", flush=True)
    orows = op_rows(cfg, S_list, mp)
    print(f"{'op':18s} {'S':>4s} {'T':>5s} {'1call_ms':>9s} {'chain_ms':>9s} "
          f"{'gpu_ms':>8s} {'cpu_us':>7s} {'L2?':>4s} {'l2_GBps':>8s}")
    for r in orows:
        print(f"{r['op']:18s} {r['S']:4d} {r['T']:5d} {r['eager_single_ms']:9.4f} "
              f"{r['eager_chain_ms']:9.4f} {r['graph_ms']:8.4f} "
              f"{r['marginal_us']:7.1f} {str(r['l2_resident']):>4s} "
              f"{r['l2_GBps']:>8}")
    with (out / "ops.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orows[0].keys()))
        w.writeheader()
        w.writerows(orows)

    print("\n=== launch cost ===", flush=True)
    lrows = launch_rows(projs, mp)
    for r in lrows:
        print(f"  {r['method']:20s} {r['us']:8.2f} us   {r['detail']}")
    with (out / "launch_cost.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lrows[0].keys()))
        w.writeheader()
        w.writerows(lrows)

    grows = []
    if not args.skip_graph:
        print("\n=== the 18-layer projection stack, eager vs CUDA graph ===", flush=True)
        grows = graph_rows(projs, layers, cfg, mp)
        for r in grows:
            if r["note"].startswith("FAILED"):
                print(f"  {r['mode']:20s} {r['note']}")
            else:
                print(f"  {r['mode']:20s} {r['median_ms']:8.3f} ms  "
                      f"{r['us_per_launch']:6.2f} us/launch  {r['note']}")
        with (out / "cuda_graph.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(grows[0].keys()))
            w.writeheader()
            w.writerows(grows)

    with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["proj", "S", "mode", "iter", "ms"])
        w.writeheader()
        w.writerows(ALL_SAMPLES)
    print(f"  wrote {len(ALL_SAMPLES)} per-iteration samples to raw/latency_samples.csv")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E2", "task": "T005",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_revision": P.REVISION,
        "model_repo": P.CHECKPOINT, "model_commit": P.REVISION, "checkpoint": P.CHECKPOINT,
        "num_denoise_steps": "not applicable — E2 measures single kernels, not a sample loop; "
                             "the policy is loaded with num_inference_steps=4 only so the "
                             "weights load identically to T002/T003",
        "action_horizon": getattr(cfg_obj, "chunk_size", None),
        "prefix_tokens": "not applicable — E2 sweeps S itself; the model's real P=816 / S=51 "
                         "are recorded in T002 (results/20260921_E3_pi0_eager_N*/meta.json)",
        "suffix_tokens": "swept: see S_swept",
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "precision": "bfloat16", "compile_mode": "none (eager)",
        "batch": 1, "n_warmup": 20, "n_timed": 100,
        "n_warmup_chain": 20, "n_timed_chain": 100, "chain_k": [1, 8, 64],
        "estimator": "median; chained rows are a 3-point least-squares fit of t(k) = fixed + k*marginal over k in {1,8,64}, with the max residual per row",
        "S_swept": S_list, "expert_geometry": cfg,
        "projections": [{"name": p["name"], "in": p["in"], "out": p["out"]} for p in projs],
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "jetson_clocks": "applied before the run; GPU locked min==max",
        "emc_hz": int(sh_out(
            "sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "peaks_used": mp,
        # protocol §5 profiler fields: this directory also carries ncu artifacts, written by
        # bench/run_ncu_validation.sh into ncu/.
        "gpu_arch": "sm_87",
        "ncu_version": sh_out("ncu --version 2>/dev/null | head -3 | tail -1"),
        "ncu_set": "explicit --metrics list (no --set)",
        "ncu_filter": '--profile-from-start off --nvtx --nvtx-include "micro/" -c 1; '
                      "cold cases use ncu's default cache control, warm cases "
                      "--cache-control none",
        "cuobjdump_version": sh_out("cuobjdump --version 2>/dev/null | tail -2 | head -1"),
        "nsys_version": None,
        "raw_files": collect_raw(out, "bench/micro_ae_kernels.py"),
        "command": f"python bench/micro_ae_kernels.py --outdir {args.outdir} --S {args.S}"
                   + (f" --layer {args.layer}" if args.layer else "")
                   + (" --nvtx" if args.nvtx else ""),
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max and recorded in `gate` before the "
                              "run; E1 established that sustained load on this device does "
                              "not move the clock.",
        "notes": [
            "Shapes are read from the loaded checkpoint, not from a table.",
            "docs/01_measurement_plan.md's E2 table lists `kv_proj (k and v)` as 1024 -> 512. "
            "The expert is multi-query (num_key_value_heads=1), so k and v are two separate "
            "1024 -> 256 GEMMs. Same weight bytes, two kernels instead of one.",
            "Floors use E1's MEASURED peaks, not the spec sheet.",
            "`bytes` counts the weight matrix once plus the input and output activations. At "
            "S = 51 the weights are ~95 % of the traffic, which is why the projections are "
            "memory-bound despite being GEMMs.",
            "The CUDA-graph stack runs the 126 projections in the real order at the real "
            "widths but omits attention and the activation function. It is a launch-count "
            "experiment, not a numerically faithful expert.",
        ],
    })
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
