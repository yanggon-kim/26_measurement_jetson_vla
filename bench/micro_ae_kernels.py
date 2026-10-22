"""T003 / E2: the π0 action expert one kernel at a time (ported from reference/orin/bench/micro_ae_kernels.py).

Shapes and weights come from the π0 checkpoint itself (the expert's layer tensors read from model.safetensors at the
pinned revision): hidden 1024, intermediate 4096, 8 query heads × 256, 1 KV head.
  q 1024→2048, k 1024→256, v 1024→256, o 2048→1024, gate / up 1024→4096, down 4096→1024

Per projection and S ∈ {1, 2, 4, 8, 16, 32, 51, 64, 128, 256}, y = F.linear(x[S, K], W[N, K]), bf16:
  kernel_ms     KERNEL time per call: CUDA-graph replay of 100-200 calls, events around the replay ÷ calls (no Python,
                no launch path). "cold": the calls cycle through distinct copies of the layers' weights, working set
                ≥ 3× the 32 MiB L2, so each call streams its weight from DRAM; "hot": one weight, resident in L2
  chain_ms      CUDA events around 1,000 calls issued back to back from Python ÷ 1,000 (the plan's "CUDA events over
                1,000 calls"): GPU-timeline time per call = max(kernel, CPU issue), since launches are asynchronous
  issue_ms      wall time per call issued back to back from Python (perf_counter around 1,000 calls + one sync)
  single_ms     one call between one event pair (carries the event / timed-region overhead; reported, not used)
plus kernel names (torch.profiler), FLOPs, bytes (weight + activations), achieved GB/s and TFLOP/s, the floor at
the MEASURED peak bandwidth / bf16 TFLOP/s of E1, and the ratio kernel_ms / floor.
Ops at the same S: SDPA (S queries, 8 heads × 256, against P + S = 816 + S keys, 1 KV head), softmax, RMSNorm, RoPE,
GeLU (tanh), add. Launch cost: an empty kernel, the smallest GEMM, and chains of k calls (fit t(k) = a + b k).
Soctherm OC counters are read around every section (bench/gate.py).

~/venvs/lerobot/bin/python bench/micro_ae_kernels.py --hw results/<E1>/hw_constants.json --out results/<run>
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402

S_LIST = [1, 2, 4, 8, 16, 32, 51, 64, 128, 256]
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
L2_BYTES = 33554432          # results/20261022_E0_platform_MAXN/hw_constants.json
SAMPLES = []


def stats(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]  # noqa: E731
    return {"median": statistics.median(v), "mean": statistics.mean(v), "p5": q(.05), "p95": q(.95), "min": v[0],
            "n": len(v)}


def ev_time(fn, reps=1, warm=20, n=30, tag=None):
    """GPU ms per call: events around `reps` calls of fn, n repetitions after warm-up."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for i in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / reps)
        if tag:
            SAMPLES.append({**tag, "iter": i, "ms_per_call": out[-1]})
    return stats(out)


def issue_time(fn, reps=1000, n=10, tag=None):
    """Wall ms per call for reps back-to-back Python-issued calls ending in one sync."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3 / reps)
        if tag:
            SAMPLES.append({**tag, "iter": i, "ms_per_call": out[-1]})
    return stats(out)


def graph_time(fn_list, n=30, tag=None):
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            for f in fn_list:
                f()
    torch.cuda.current_stream().wait_stream(st)
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    r = ev_time(lambda: g.replay(), reps=1, warm=10, n=n)
    k = len(fn_list)
    if tag:
        SAMPLES.append({**tag, "iter": -1, "ms_per_call": r["median"] / k})
    del g
    return {key: (v / k if key != "n" else v) for key, v in r.items()}


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def load_expert(ckpt_dir):
    from safetensors import safe_open
    f = safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt", device="cpu")
    keys = [k for k in f.keys() if "gemma_expert" in k and ".layers." in k and k.endswith(".weight")]
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in keys})
    W = {p: [] for p in PROJ}
    for li in layers:
        for p in PROJ:
            k = next(k for k in keys if f".layers.{li}." in k and k.endswith(f"{p}.weight"))
            W[p].append(f.get_tensor(k).to("cuda", torch.bfloat16))
    return W, len(layers)


def proj_rows(W, nl, peaks):
    rows = []
    bw, fl = peaks["bw_Bps"], peaks["bf16_flops"]
    for p in PROJ:
        N, K = W[p][0].shape
        wb = N * K * 2
        copies = max(nl, -(-3 * L2_BYTES // wb))            # ≥ 3× L2 working set
        ws = [W[p][i % nl] if i < nl else W[p][i % nl].clone() for i in range(copies)]
        for S in S_LIST:
            x = torch.randn(S, K, device="cuda", dtype=torch.bfloat16)
            it = {"i": 0}

            def cold(x=x, ws=ws, it=it):
                w = ws[it["i"]]
                it["i"] = (it["i"] + 1) % len(ws)
                return F.linear(x, w)

            def hot(x=x, w=W[p][0]):
                return F.linear(x, w)

            tag = {"section": "proj", "op": p, "S": S}
            names = kernel_names(hot)
            oc0 = gate.oc_counts()
            ch = ev_time(cold, reps=1000, warm=50, n=10, tag={**tag, "mode": "chain_event_cold_1000"})
            iss = issue_time(cold, tag={**tag, "mode": "issue_cold_1000"})
            n_g = min(max(100, len(ws)), 200)          # the replay touches every copy: working set ≥ 3× L2
            kc = graph_time([lambda x=x, w=ws[i % len(ws)]: F.linear(x, w) for i in range(n_g)],
                            tag={**tag, "mode": f"graph_cold_{n_g}"})
            kh = graph_time([hot] * 100, tag={**tag, "mode": "graph_hot_100"})
            sg = ev_time(hot, reps=1, warm=20, n=100, tag={**tag, "mode": "single_event"})
            oc1 = gate.oc_counts()
            flops = 2 * S * K * N
            nb = wb + 2 * S * (K + N)
            floor = max(nb / bw, flops / fl) * 1e3
            rows.append({"op": p, "S": S, "K": K, "N": N, "kernel_names": " | ".join(names),
                         "kernel_cold_ms": round(kc["median"], 6), "kernel_cold_p5_ms": round(kc["p5"], 6),
                         "kernel_cold_p95_ms": round(kc["p95"], 6), "kernel_hot_ms": round(kh["median"], 6),
                         "chain_ms": round(ch["median"], 6), "chain_p95_ms": round(ch["p95"], 6),
                         "issue_ms": round(iss["median"], 6), "issue_p95_ms": round(iss["p95"], 6),
                         "single_event_ms": round(sg["median"], 6),
                         "single_event_p95_ms": round(sg["p95"], 6),
                         "regime": "cpu_bound (issue > 1.15 × kernel)" if iss["median"] > 1.15 * kc["median"] else "gpu_bound",
                         "flops": flops, "weight_bytes": wb, "act_bytes": 2 * S * (K + N), "bytes": nb,
                         "cold_working_set_bytes": wb * len(ws), "intensity_flop_per_B": round(flops / nb, 3),
                         "achieved_GBps": round(nb / (kc["median"] / 1e3) / 1e9, 2),
                         "achieved_TFLOPps": round(flops / (kc["median"] / 1e3) / 1e12, 4),
                         "floor_ms": round(floor, 6), "floor_bound": "memory" if nb / bw >= flops / fl else "compute",
                         "kernel_over_floor": round(kc["median"] / floor, 3),
                         "issue_over_floor": round(iss["median"] / floor, 3),
                         "oc_event_delta": sum(oc1[k] - oc0[k] for k in oc0), "n_kernel": kc["n"]})
            print(f"{p} S={S}: kernel {kc['median']*1e3:.1f} us (hot {kh['median']*1e3:.1f}), issue "
                  f"{iss['median']*1e3:.1f} us, chain {ch['median']*1e3:.1f} us, x floor {rows[-1]['kernel_over_floor']}",
                  flush=True)
            del x
        del ws
        torch.cuda.empty_cache()
    return rows


def op_rows(peaks, P=816):
    rows = []
    H, D, Hn, I = 8, 256, 1024, 4096
    dt = torch.bfloat16
    for S in S_LIST:
        T = P + S
        q = torch.randn(1, H, S, D, device="cuda", dtype=dt)
        k = torch.randn(1, 1, T, D, device="cuda", dtype=dt)
        v = torch.randn(1, 1, T, D, device="cuda", dtype=dt)
        h = torch.randn(S, Hn, device="cuda", dtype=dt)
        h2 = torch.randn(S, Hn, device="cuda", dtype=dt)
        mlp = torch.randn(S, I, device="cuda", dtype=dt)
        sc = torch.randn(1, H, S, T, device="cuda", dtype=dt)
        ro = torch.randn(1, H, S, D, device="cuda", dtype=dt)
        cos = torch.randn(1, 1, S, D, device="cuda", dtype=dt)
        sin = torch.randn(1, 1, S, D, device="cuda", dtype=dt)
        nw = torch.randn(Hn, device="cuda", dtype=dt)

        def rms(h=h, w=nw):
            x = h.float()
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dt) * w

        def rope(ro=ro, cos=cos, sin=sin):
            x1, x2 = ro[..., : D // 2], ro[..., D // 2:]
            return ro * cos + torch.cat((-x2, x1), dim=-1) * sin

        ops = [("sdpa_attention", lambda q=q, k=k, v=v: F.scaled_dot_product_attention(
                    q, k.expand(1, H, T, D), v.expand(1, H, T, D)),
                2 * S * H * D * 2 + 2 * T * D * 2, 4 * H * S * T * D, f"S={S} x T={T}, 1 KV head (expand view)"),
               ("softmax_fp32", lambda sc=sc: torch.softmax(sc.float(), dim=-1), H * S * T * (2 + 4), 0, f"[1,{H},{S},{T}]"),
               ("rmsnorm", rms, 2 * S * Hn * 2 + Hn * 2, 0, f"[{S},{Hn}]"),
               ("rope", rope, 2 * H * S * D * 2, 0, f"[1,{H},{S},{D}]"),
               ("gelu_tanh", lambda mlp=mlp: F.gelu(mlp, approximate="tanh"), 2 * S * I * 2, 0, f"[{S},{I}]"),
               ("add", lambda h=h, h2=h2: h + h2, 3 * S * Hn * 2, 0, f"[{S},{Hn}]")]
        for name, fn, nb, flops, note in ops:
            tag = {"section": "op", "op": name, "S": S}
            names = kernel_names(fn)
            ch = ev_time(fn, reps=1000, warm=50, n=10, tag={**tag, "mode": "chain_event_1000"})
            iss = issue_time(fn, tag={**tag, "mode": "issue_1000"})
            try:
                kt = graph_time([fn] * 100, tag={**tag, "mode": "graph_100"})
            except Exception as exc:
                kt = None
                cap_err = f"graph capture failed: {type(exc).__name__}: {str(exc)[:120]}"
            rows.append({"op": name, "S": S, "T": T, "kernel_names": " | ".join(names),
                         "kernel_ms": round(kt["median"], 6) if kt else "", "kernel_p95_ms": round(kt["p95"], 6) if kt else "",
                         "chain_ms": round(ch["median"], 6), "issue_ms": round(iss["median"], 6),
                         "regime": ("cpu_bound (issue > 1.15 × kernel)" if kt and iss["median"] > 1.15 * kt["median"]
                                    else "gpu_bound") if kt else "n/a",
                         "bytes_touched": nb, "flops": flops, "working_set_fits_L2": nb < L2_BYTES,
                         "l2_or_dram_GBps": round(nb / (kt["median"] / 1e3) / 1e9, 2) if kt else "",
                         "note": note + ("; working set < L2: GB/s is cache bandwidth, not DRAM" if nb < L2_BYTES else "")
                                 + ("" if kt else "; " + cap_err)})
            print(f"{name} S={S}: kernel {kt['median']*1e3 if kt else float('nan'):.1f} us, issue "
                  f"{iss['median']*1e3:.1f} us", flush=True)
        del q, k, v, h, h2, mlp, sc, ro, cos, sin, nw
        torch.cuda.empty_cache()
    return rows


def launch_rows(W):
    rows = []
    e = torch.zeros(1, device="cuda")
    for name, fn in [("empty_kernel_add_", lambda: e.add_(1.0)),
                     ("smallest_gemm_q_proj_S1", lambda x=torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16),
                      w=W["q_proj"][0]: F.linear(x, w))]:
        kt = ev_time(fn, reps=1000, warm=50, n=10)
        iss = issue_time(fn, reps=1000, n=10)
        sg = ev_time(fn, reps=1, warm=50, n=500)
        rows.append({"method": name, "gpu_us_per_call": round(kt["median"] * 1e3, 3),
                     "issue_us_per_call": round(iss["median"] * 1e3, 3),
                     "single_event_pair_us": round(sg["median"] * 1e3, 3),
                     "note": "issue = CPU launch cost per call when the GPU work is shorter (asynchronous launches)"})
    x = torch.randn(51, 1024, device="cuda", dtype=torch.bfloat16)
    w = W["q_proj"][0]
    pts = []
    for k in (1, 8, 64, 512):
        r = ev_time(lambda k=k: [F.linear(x, w) for _ in range(k)], reps=1, warm=10, n=50)
        pts.append((k, r["median"]))
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    a = my - b * mx
    rows.append({"method": "chain_fit_q_proj_S51", "gpu_us_per_call": round(b * 1e3, 3),
                 "issue_us_per_call": "", "single_event_pair_us": round(a * 1e3, 3),
                 "note": f"t(k) = a + b·k over k = 1, 8, 64, 512 within one event pair: b per call, a fixed; points {pts}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    ck = snapshot_download("lerobot/pi0_libero_finetuned_v044", revision="45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
                           local_files_only=True)
    hw = json.load(open(a.hw))
    peaks = {"bw_Bps": hw["measured_bw_GBps"]["best"]["value"] * 1e9,
             "bf16_flops": hw["measured_tflops"]["bf16"]["value"] * 1e12, "source": a.hw}
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    g0 = gate.run_gate()
    if not g0["gate_pass"]:
        raise SystemExit(f"gate failed {g0['checks']}")
    W, nl = load_expert(ck)
    shapes = {p: list(W[p][0].shape) for p in PROJ}
    print("expert layers", nl, shapes, flush=True)
    t0 = time.time()
    proj = proj_rows(W, nl, peaks)
    ops = op_rows(peaks)
    lau = launch_rows(W)
    g1 = gate.run_gate(apply_clocks=False, check_idle=False)
    for name, rows in (("kernels.csv", proj), ("ops.csv", ops), ("launch_cost.csv", lau)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(a.out, "raw", "samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "op", "S", "mode", "iter", "ms_per_call"])
        w.writeheader()
        w.writerows(SAMPLES)
    json.dump({"gate_before": g0, "gate_after": g1, "wall_s": time.time() - t0, "expert_layers": nl,
               "expert_shapes": shapes, "peaks": peaks, "checkpoint": ck,
               "oc_delta_total": {k: g1["soctherm_oc_event_cnt"][k] - g0["soctherm_oc_event_cnt"][k]
                                  for k in g0["soctherm_oc_event_cnt"]}},
              open(os.path.join(a.out, "raw", "run.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
