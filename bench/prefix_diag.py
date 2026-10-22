"""T005 diagnostic: why is pi0's VLM prefix slower than pi0.5's (fewer tokens, same Gemma-2B, same code path)?

Loads one model in a named rung (bench/rungs.py, default eager_bf16: bf16 weights, nothing compiled), captures the
exact arguments of the prefill call (<container>.forward with inputs_embeds[1] is None) from one real inference, then
replays ONLY that call: CUDA-event timing (warm-up 10, timed 50) and one torch.profiler pass (5 calls) for the per-op
and per-kernel CUDA time with input shapes. Records the dtype / shape / stride of every argument, of the attention
mask, and of every Linear weight in the language model. Writes <out>/prefix_<model>.json + prefix_<model>_ops.csv
+ prefix_<model>_kernels.csv.

python bench/prefix_diag.py --model pi0 --out results/<dir>
"""
import argparse
import csv
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def describe(x):
    if torch.is_tensor(x):
        return {"shape": list(x.shape), "dtype": str(x.dtype), "stride": list(x.stride()), "contiguous": x.is_contiguous(),
                "min": float(x.float().min()) if x.numel() and x.dtype != torch.bool else None}
    if isinstance(x, (list, tuple)):
        return [describe(v) for v in x]
    if isinstance(x, dict):
        return {k: describe(v) for k, v in x.items()}
    return repr(x)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rung", default="eager_bf16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import gate as G
    import rungs as RG
    import run_e3 as E3
    os.makedirs(a.out, exist_ok=True)
    g0 = G.wait_for_gate(apply_clocks=True)
    ad = E3.make_adapter(a.model)
    ad.name, ad.family = a.model, "lerobot"
    desc = RG.apply_rung(ad, a.rung)
    pg = getattr(ad.policy.model, ad.spec.container)
    cap = {}
    orig = pg.forward

    def grab(*args, **kw):
        emb = kw.get("inputs_embeds")
        if emb is not None and emb[1] is None and "kw" not in cap:
            cap["kw"] = kw
        return orig(*args, **kw)
    pg.forward = grab
    ad.iteration()
    pg.forward = orig
    kw = cap["kw"]
    lm = pg.paligemma.model.language_model
    weights = {}
    for n, m in lm.named_modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight
            k = f"{tuple(w.shape)} {w.dtype} contiguous={w.is_contiguous()} stride={tuple(w.stride())}"
            weights[k] = weights.get(k, 0) + 1
    info = {"model": a.model, "rung": a.rung, "rung_desc": desc, "gate": {k: g0[k] for k in ("time_utc", "checks", "gpu_hz")},
            "prefill_kwargs": describe({k: v for k, v in kw.items() if k != "past_key_values"}),
            "attn_implementation_seen": lm.config._attn_implementation,
            "lm_linear_weights": weights,
            "lm_config": {k: getattr(lm.config, k, None) for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                                                     "num_attention_heads", "num_key_value_heads", "head_dim",
                                                                     "hidden_activation", "torch_dtype", "dtype")}}

    def call():
        with torch.inference_mode():
            return pg.forward(**kw)
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    info["prefill_replay_ms"] = {"median": statistics.median(ts), "min": min(ts), "max": max(ts), "n": len(ts)}
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(5):
            call()
        torch.cuda.synchronize()
    ops = []
    for ev in prof.key_averages(group_by_input_shape=True):
        dev = getattr(ev, "device_time_total", getattr(ev, "cuda_time_total", 0))
        if dev > 0:
            ops.append({"op": ev.key, "input_shapes": str(ev.input_shapes)[:300], "calls": ev.count,
                        "device_time_ms_per_call_total": dev / 1e3 / 5})
    ops.sort(key=lambda r: -r["device_time_ms_per_call_total"])
    kern = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            d = kern.setdefault(ev.name, [0, 0.0])
            d[0] += 1
            d[1] += ev.device_time if hasattr(ev, "device_time") else ev.cuda_time
    krows = sorted(({"kernel": k, "count_per_call": v[0] / 5, "ms_per_call": v[1] / 1e3 / 5} for k, v in kern.items()),
                   key=lambda r: -r["ms_per_call"])
    info["kernel_ms_per_call_total"] = sum(r["ms_per_call"] for r in krows)
    for name, rows in ((f"prefix_{a.model}_ops.csv", ops), (f"prefix_{a.model}_kernels.csv", krows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump(info, open(os.path.join(a.out, f"prefix_{a.model}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: info[k] for k in ("prefill_replay_ms", "kernel_ms_per_call_total", "attn_implementation_seen",
                                           "lm_linear_weights")}, indent=1, default=str))
    for r in krows[:12]:
        print(f"{r['ms_per_call']:8.2f} ms  x{r['count_per_call']:6.0f}  {r['kernel'][:110]}")


if __name__ == "__main__":
    main()
