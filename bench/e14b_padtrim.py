"""T017 (E14b): pi0 prefix padding trim, decomposed -- pad / nocam / trimprompt / nopad.

Variants are bench/pi0_padtrim.py (harness-level; LeRobot untouched, shared with T016 step 3's `nopad`):
the empty camera is removed from the image list BEFORE the image encoder (no SigLIP / projector call, no
tokens, no mask entry -- LeRobot's own `_preprocess_images` still builds its -1 dummy tensor, outside every
timed / profiled region, and it is dropped unused); the prompt is sliced to its valid tokens.

Modes (one process each; MAXN only):
  gate    Step 1. Shipped model (checkpoint dtype, eager, E8 `stock`, matmul "highest"), the 64 T017 inputs
          (bench/fixtures/e14b, noise seed 17000 + i), each variant vs pad: REL <= 1e-4 on every input or
          the gate fails (exit 5). Saves raw/actions_<variant>_fp32.npz (pad fp32 = the floor's `stock`).
  equiv   Step 3. Compiled bf16_both rung (as E7), automatic_dynamic_shapes OFF (one static compile per
          prefix length; compiles counted and timed), same inputs and noise: pad, each variant, then a
          second pad pass in the same process (control i). Saves raw/actions_<variant>_bf16.npz.
  verdict Step 3 analysis over the saved .npz: equiv.csv (fp32 gate, bf16 variant vs pad, floor stock vs
          bf16 pad, pad repeat), verdicts.csv (bitwise / PASS / FAIL, pre-registered), summary.json.
  time    Step 2a + 2c. E12b graphs_both harness per variant (bench/e12b_corun.Harness: owngraph_prefix +
          memo_681 + owngraph_expert + dbuf_kv, whole GPU, prefix_first), plus `split_vision`: a vision graph
          (every camera's embed_image, projector included) and a vlm_prefix graph (LeRobot's embed_prefix +
          Gemma prefill with embed_image answered from the vision graph's outputs), captured per variant.
          Blocks pad > nocam > trimprompt > nopad > pad > nopad > trimprompt > nocam > pad, >= 60 frames of
          prefix / expert / serial / split per block, state per block; a second bracket if pad drifts > 1 %.
          K/V handover bytes per variant (unique cache tensors' nbytes + shapes).
  ncu     Step 2b target for bench/run_ncu_e14b.sh: NVTX <variant>_{prefix,expert,vision,vlm}, 5 frames each.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import pi0_common as P           # noqa: E402
import pi0_padtrim as PT         # noqa: E402

N = 10
V = ("pad", "nocam", "trimprompt", "nopad")
BRACKET = ("pad", "nocam", "trimprompt", "nopad", "pad", "nopad", "trimprompt", "nocam", "pad")
INPUTS = REPO / "bench/fixtures/e14b"
SEED0 = 17_000
KV_EXPECTED = {"pad": 15_040_512, "nocam": 10_321_920, "trimprompt": 14_542_848, "nopad": 9_824_256}


def static_shapes():
    import torch._dynamo as dyn                                                   # noqa: PLC0415
    dyn.config.automatic_dynamic_shapes = False
    dyn.config.cache_size_limit = max(64, dyn.config.cache_size_limit)


def dyn_counters():
    import torch._dynamo.utils as du                                              # noqa: PLC0415
    return {k: dict(v) for k, v in du.counters.items() if k in ("stats", "frames")}


def state() -> dict:
    g = C.gate_full()
    g["temps_mC"] = C.temps()
    g["t_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return g


def env_dim(policy) -> int:
    from lerobot.utils.constants import ACTION                                    # noqa: PLC0415
    return int(policy.config.output_features[ACTION].shape[0])


# ------------------------------------------------------------------------------------------------ actions
def run_actions(policy, spec, frames, variants, fin, tag, out, extra_pass=None):
    """actions[variant] = (64, chunk, max_action_dim) float32 on the host; compile wall per (variant, L)."""
    acts, first_call = {v: [] for v in variants}, {}
    with torch.no_grad():
        for fr in frames:
            for v in variants:
                ctx = PT.ctx_for(fr["ctx"], v)
                key = f"{v}@L{ctx['lang'].shape[1]}"
                t0 = time.perf_counter()
                a = PT.sample(policy, ctx, fr["noise"])
                fin.add(a)
                if key not in first_call:
                    torch.cuda.synchronize()
                    first_call[key] = round(time.perf_counter() - t0, 3)
                acts[v].append(a.float().cpu().numpy())
        if extra_pass:
            acts[extra_pass] = []
            for fr in frames:
                a = PT.sample(policy, PT.ctx_for(fr["ctx"], "pad"), fr["noise"])
                fin.add(a)
                acts[extra_pass].append(a.float().cpu().numpy())
    seeds = np.array([fr["seed"] for fr in frames])
    for v, lst in acts.items():
        np.savez(out / "raw" / f"actions_{v}_{tag}.npz", actions=np.stack(lst), seeds=seeds,
                 obs=np.array([fr["obs_id"] for fr in frames]))
    return acts, first_call


def cmp(a, b, d):
    """a, b: (1, chunk, D) arrays; metrics on the first d dims (primary: the env's) and on all."""
    out = {}
    for lab, sl in (("env", slice(0, d)), ("all", slice(None))):
        x, y = a[..., sl].astype(np.float64), b[..., sl].astype(np.float64)
        diff = y - x
        out[f"maxabs_{lab}"] = float(np.abs(diff).max())
        out[f"rel_{lab}"] = float(np.linalg.norm(diff) / np.linalg.norm(x))
        out[f"bitwise_{lab}"] = bool(np.array_equal(a[..., sl], b[..., sl]))
    return out


def mode_gate(a, out):
    """Step 1. Two passes over the same inputs and noise, same process:
    (1) the SHIPPED model as loaded (checkpoint dtype -- this checkpoint stores part of its parameters in bf16),
        eager: `stock`. Its variant-vs-pad REL is reported (first gate attempt, 2026-09-27: REL ~2e-3 = bf16
        arithmetic, not a trim defect), and its pad actions are the noise floor's `stock` side (Step 3);
    (2) the same model cast to fp32 (`policy.float()`, TF32 off, matmul "highest"), eager: THE fp32 gate,
        REL <= 1e-4 for every input and variant."""
    import model_registry as MR                                                   # noqa: PLC0415
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=N, compile_mode=None)
    P.assert_real_weights(policy, spec=spec)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dt = {}
    for p_ in policy.parameters():
        dt[str(p_.dtype)] = dt.get(str(p_.dtype), 0) + p_.numel()
    frames, manifest = CR.load_frames(policy, spec, INPUTS, a.n_inputs, seed0=SEED0)
    fin = C.Finite()
    d = env_dim(policy)
    stock, first_stock = run_actions(policy, spec, frames, V, fin, "stock", out)
    policy.float()
    dt32 = sorted({str(p_.dtype) for p_ in policy.parameters()} | {str(b_.dtype) for b_ in policy.buffers()
                                                                     if b_.is_floating_point()})
    acts, first = run_actions(policy, spec, frames, V, fin, "fp32", out)
    rows = []
    for tag, A in (("checkpoint_dtype_attempt", stock), ("fp32_gate", acts)):
        for v in V[1:]:
            for i in range(len(frames)):
                rows.append({"input": i, "comparison": tag, "variant": v, "vs": "pad", **cmp(A["pad"][i], A[v][i], d)})
    worst = {v: max(r["rel_env"] for r in rows if r["variant"] == v and r["comparison"] == "fp32_gate") for v in V[1:]}
    worst_all = {v: max(r["rel_all"] for r in rows if r["variant"] == v and r["comparison"] == "fp32_gate") for v in V[1:]}
    worst_stock = {v: max(r["rel_env"] for r in rows if r["variant"] == v and r["comparison"] == "checkpoint_dtype_attempt")
                   for v in V[1:]}
    nf = fin.read()
    passed = all(x <= 1e-4 for x in worst.values()) and nf == 0
    with (out / "gate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    res = {"gate": "PASS" if passed else "FAIL", "threshold_rel": 1e-4, "worst_rel_env": worst,
           "worst_rel_all": worst_all, "checkpoint_dtype_attempt_worst_rel_env": worst_stock,
           "checkpoint_param_numel_by_dtype": dt, "gate_dtypes_after_cast": dt32, "nonfinite": nf,
           "inputs": len(frames), "first_call_s": {"stock": first_stock, "fp32": first}}
    (out / "gate_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    return res, spec, {"stock": "shipped model as loaded (checkpoint dtype), eager",
                       "gate": "same model cast to fp32 (policy.float()), eager, TF32 off, matmul highest",
                       "checkpoint_param_numel_by_dtype": dt}, manifest


def mode_equiv(a, out):
    static_shapes()
    policy, spec, cfg, note = C.load_rung(N)
    frames, manifest = CR.load_frames(policy, spec, INPUTS, a.n_inputs, seed0=SEED0)
    fin = C.Finite()
    c0 = dyn_counters()
    t0 = time.perf_counter()
    acts, first = run_actions(policy, spec, frames, V, fin, "bf16", out, extra_pass="pad_rep")
    wall = round(time.perf_counter() - t0, 1)
    nf = fin.read()
    tok = {v: PT.prefix_tokens(policy, PT.ctx_for(frames[0]["ctx"], v)) for v in V}
    res = {"inputs": len(frames), "nonfinite": nf, "first_call_s": first, "wall_s": wall,
           "dynamo_counters_before": c0, "dynamo_counters_after": dyn_counters(), "tokens_fixture": tok,
           "action_shape": list(acts["pad"][0].shape), "env_action_dims": env_dim(policy)}
    (out / "equiv_run.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("nonfinite", "wall_s", "first_call_s")}, indent=1), flush=True)
    return res, spec, note, manifest


def mode_verdict(a, out):
    L = lambda v, t: np.load(out / "raw" / f"actions_{v}_{t}.npz")                 # noqa: E731
    d = int(json.loads((out / "equiv_run.json").read_text())["env_action_dims"])
    rows = []
    fp = {v: L(v, "fp32") for v in V}
    sk = {v: L(v, "stock") for v in V}
    bf = {v: L(v, "bf16") for v in (*V, "pad_rep")}
    for z in (*fp.values(), *sk.values(), *bf.values()):
        assert np.array_equal(z["seeds"], fp["pad"]["seeds"]), "seed sets differ"
    n = len(fp["pad"]["actions"])
    for i in range(n):
        for v in V[1:]:
            rows.append({"input": i, "comparison": "fp32_gate", "variant": v, "vs": "pad",
                         **cmp(fp["pad"]["actions"][i], fp[v]["actions"][i], d)})
            rows.append({"input": i, "comparison": "checkpoint_dtype_attempt", "variant": v, "vs": "pad",
                         **cmp(sk["pad"]["actions"][i], sk[v]["actions"][i], d)})
            rows.append({"input": i, "comparison": "bf16_variant_vs_pad", "variant": v, "vs": "pad",
                         **cmp(bf["pad"]["actions"][i], bf[v]["actions"][i], d)})
        rows.append({"input": i, "comparison": "floor_stock_vs_bf16", "variant": "pad", "vs": "stock_pad",
                     **cmp(sk["pad"]["actions"][i], bf["pad"]["actions"][i], d)})
        rows.append({"input": i, "comparison": "control_pad_repeat", "variant": "pad", "vs": "pad",
                     **cmp(bf["pad"]["actions"][i], bf["pad_rep"]["actions"][i], d)})
    with (out / "equiv.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def agg(sel, lab="env"):
        R = [r for r in rows if sel(r)]
        return {"n": len(R), "bitwise": sum(r[f"bitwise_{lab}"] for r in R),
                "MAXABS": max(r[f"maxabs_{lab}"] for r in R), "REL": max(r[f"rel_{lab}"] for r in R),
                "median_maxabs": statistics.median(r[f"maxabs_{lab}"] for r in R),
                "median_rel": statistics.median(r[f"rel_{lab}"] for r in R)}
    summ = {"env_action_dims": d}
    for lab in ("env", "all"):
        F = agg(lambda r: r["comparison"] == "floor_stock_vs_bf16", lab)
        summ[f"floor_{lab}"] = F
        summ[f"control_pad_repeat_{lab}"] = agg(lambda r: r["comparison"] == "control_pad_repeat", lab)
        for v in V[1:]:
            summ[f"gate_{v}_{lab}"] = agg(lambda r, v=v: r["comparison"] == "fp32_gate" and r["variant"] == v, lab)
            summ[f"checkpoint_dtype_{v}_{lab}"] = agg(
                lambda r, v=v: r["comparison"] == "checkpoint_dtype_attempt" and r["variant"] == v, lab)
            summ[f"bf16_{v}_{lab}"] = agg(lambda r, v=v: r["comparison"] == "bf16_variant_vs_pad" and r["variant"] == v, lab)
    verd = []
    for v in V[1:]:
        s, F = summ[f"bf16_{v}_env"], summ["floor_env"]
        if s["bitwise"] == s["n"]:
            vd = "bitwise"
        elif s["MAXABS"] <= F["MAXABS"] and s["REL"] <= F["REL"]:
            vd = "PASS"
        else:
            vd = "FAIL"
        verd.append({"variant": v, "verdict": vd, "bitwise": f"{s['bitwise']}/{s['n']}", "MAXABS": s["MAXABS"],
                     "REL": s["REL"], "F_maxabs": F["MAXABS"], "F_rel": F["REL"],
                     "gate_REL_fp32": summ[f"gate_{v}_env"]["REL"], "primary_dims": f"env ({d})"})
    with (out / "verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(verd[0])); w.writeheader(); w.writerows(verd)
    summ["verdicts"] = verd
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    for r in verd:
        print(r, flush=True)
    return summ


# ------------------------------------------------------------------------------------------------ timing
class Split:
    """split_vision for one variant, on the harness's own static prefix inputs (h.pctx)."""

    def __init__(self, h):
        self.h = h
        pwe = h.m.paligemma_with_expert
        self.pool = torch.cuda.graph_pool_handle()
        imgs = h.pctx["images"]
        with torch.no_grad(), h._on_p():                                          # noqa: SLF001
            for _ in range(3):
                [pwe.embed_image(t) for t in imgs]
            torch.cuda.synchronize()
            self.gv = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.gv, pool=self.pool, stream=torch.cuda.current_stream()):
                self.vout = [pwe.embed_image(t) for t in imgs]
            torch.cuda.synchronize()
            real = pwe.embed_image
            it = {"i": 0}

            def fed(_img):
                o = self.vout[it["i"] % len(self.vout)]
                it["i"] += 1
                return o
            pwe.embed_image = fed
            try:
                for _ in range(3):
                    h._prefix_eager(h.pctx)                                       # noqa: SLF001
                torch.cuda.synchronize()
                self.gl = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.gl, pool=self.pool, stream=torch.cuda.current_stream()):
                    self.lout = h._prefix_eager(h.pctx)                           # noqa: SLF001
                torch.cuda.synchronize()
            finally:
                pwe.embed_image = real

    def run(self, F):
        rec = []
        ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(F)]
        with torch.no_grad(), self.h._on_p():                                     # noqa: SLF001
            s = torch.cuda.current_stream()
            for k in range(F):
                self.h._load_prefix_inputs(self.h.frames[k % len(self.h.frames)]["ctx"])  # noqa: SLF001
                ev[k][0].record(s)
                self.gv.replay()
                ev[k][1].record(s)
                self.gl.replay()
                ev[k][2].record(s)
        torch.cuda.synchronize()
        for k in range(F):
            v, p = ev[k][0].elapsed_time(ev[k][1]), ev[k][1].elapsed_time(ev[k][2])
            rec.append({"vision_ms": v, "vlm_prefix_ms": p, "split_sum_ms": v + p})
        return rec


def kv_bytes(h):
    seen, tot, shapes = set(), 0, []
    for t in H.leaves(h.kv[0]):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        tot += t.numel() * t.element_size()
        shapes.append(list(t.shape))
    return tot, shapes


def build_harnesses(frames, policy, spec):
    hs, sp = {}, {}
    with torch.no_grad():
        for v in V:
            fr = [{**f, "ctx": PT.ctx_for(f["ctx"], v)} for f in frames]
            t0 = time.perf_counter()
            hs[v] = H.Harness(policy, spec, fr, N, config="graphs_both", order="prefix_first")
            sp[v] = Split(hs[v])
            torch.cuda.synchronize()
            print(f"  {v}: harness + split built in {time.perf_counter() - t0:.1f} s ({hs[v].label()})", flush=True)
    return hs, sp


def mode_time(a, out):
    static_shapes()
    policy, spec, cfg, note = C.load_rung(N)
    note["expert"] = H.recompile_default(policy, spec, "denoise_step")
    note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
    note["memo"] = H.memo_681(policy)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    tok = {v: PT.prefix_tokens(policy, PT.ctx_for(frames[0]["ctx"], v)) for v in V}
    c0 = dyn_counters()
    t0 = time.perf_counter()
    hs, sp = build_harnesses(frames, policy, spec)
    build_s = round(time.perf_counter() - t0, 1)
    fin = C.Finite()
    with torch.no_grad():
        for v in V:
            for c in ("prefix_alone", "expert_alone", "serial"):
                hs[v].run(c, 6, fin)
            sp[v].run(6)
    kv = []
    for v in V:
        b, sh = kv_bytes(hs[v])
        kv.append({"variant": v, "prefix_tokens": tok[v]["prefix_tokens"],
                   "measured_bytes": b, "expected_bytes": KV_EXPECTED[v], "match": b == KV_EXPECTED[v],
                   "tensors": len(sh), "shapes": json.dumps(sh[:2]) + (f" ... x{len(sh)}" if len(sh) > 2 else "")})
    blocks, samples = [], []

    def block(bi, v):
        st0 = state()
        with torch.no_grad():
            for c, key in (("prefix_alone", "prefix_ms"), ("expert_alone", "expert_ms"), ("serial", "frame_ms")):
                r = hs[v].run(c, a.frames, fin)
                vals = [x[key] for x in r["records"]]
                stage = {"prefix_ms": "prefix", "expert_ms": "expert", "frame_ms": "serial"}[key]
                blocks.append({"block": bi, "variant": v, "stage": stage, **C.summ(vals)})
                samples.extend({"block": bi, "variant": v, "stage": stage, "frame": j, "ms": round(x, 5)}
                               for j, x in enumerate(vals))
            r = sp[v].run(a.frames)
            for key, stage in (("vision_ms", "vision"), ("vlm_prefix_ms", "vlm_prefix"), ("split_sum_ms", "split_sum")):
                vals = [x[key] for x in r]
                blocks.append({"block": bi, "variant": v, "stage": stage, **C.summ(vals)})
                samples.extend({"block": bi, "variant": v, "stage": stage, "frame": j, "ms": round(x, 5)}
                               for j, x in enumerate(vals))
        st1 = state()
        for b in blocks:
            if b["block"] == bi and "gpu_hz_start" not in b:
                b.update({"gpu_hz_start": st0["gpu_cur_freq_hz"], "gpu_hz_end": st1["gpu_cur_freq_hz"],
                          "emc_hz": st0.get("emc_rate_hz"), "temp_max_C_end": round(max(st1["temps_mC"].values()) / 1000, 1),
                          "throttled": st1["gpu_cur_freq_hz"] != st0["gpu_cur_freq_hz"], "t_utc": st1["t_utc"]})
        pm = next(b["median_ms"] for b in blocks if b["block"] == bi and b["stage"] == "prefix")
        print(f"  block {bi} {v:10s} prefix {pm:8.3f} ms", flush=True)

    order = list(BRACKET)
    for bi, v in enumerate(order):
        block(bi, v)
    padp = [b["median_ms"] for b in blocks if b["variant"] == "pad" and b["stage"] == "prefix"]
    drift = (max(padp) - min(padp)) / min(padp)
    extra = drift > 0.01
    if extra:
        print(f"  pad prefix drift {drift:.3%} > 1 %: one more bracket", flush=True)
        for v in BRACKET:
            block(len(order), v)
            order.append(v)
    nf = fin.read()
    stages = []
    for v in V:
        for stage in ("vision", "vlm_prefix", "split_sum", "prefix", "expert", "serial"):
            vals = [s["ms"] for s in samples if s["variant"] == v and s["stage"] == stage]
            stages.append({"variant": v, "stage": stage, **C.summ(vals)})
    for name, rows in (("stages.csv", stages), ("blocks.csv", blocks), ("kv_handover.csv", kv),
                       ("raw/latency_samples.csv", samples)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    med = {(r["variant"], r["stage"]): r["median_ms"] for r in stages}
    rho = {f"{st}_{v}": round(med[(v, st)] / med[("pad", st)], 4) for v in V[1:]
           for st in ("vision", "vlm_prefix", "prefix", "expert", "serial")}
    res = {"nonfinite": nf, "pad_prefix_drift": round(drift, 5), "extra_bracket": extra, "blocks": order,
           "rho": rho, "tokens": tok, "build_s": build_s, "dynamo_counters_before": c0,
           "dynamo_counters_after": dyn_counters(), "kv": kv,
           "reference_E12_whole_gpu": {"prefix": 246.79, "expert": 101.48, "serial": 348.73,
                                       "source": "results/20260925_E12_pi0_compiled_stages_N10_MAXN"}}
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"rho": rho, "drift": drift, "nonfinite": nf}, indent=1), flush=True)
    return res, spec, note, manifest, hs["pad"].label()


def mode_ncu(a):
    static_shapes()
    policy, spec, cfg, note = C.load_rung(N)
    H.recompile_default(policy, spec, "denoise_step")
    H.recompile_default(policy, spec, "embed_image")
    H.memo_681(policy)
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    hs, sp = build_harnesses(frames, policy, spec)
    fin = C.Finite()
    with torch.no_grad():
        for v in V:
            for c in ("prefix_alone", "expert_alone", "serial"):
                hs[v].run(c, 6, fin)
            sp[v].run(6)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for v in V:
            for c in ("prefix", "expert"):
                torch.cuda.nvtx.range_push(f"{v}_{c}")
                hs[v].run(f"{c}_alone", a.frames, fin)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            # split: vision graph and vlm graph under their own ranges, a.frames each
            for k in range(a.frames):
                with torch.no_grad(), hs[v]._on_p():                              # noqa: SLF001
                    hs[v]._load_prefix_inputs(hs[v].frames[k % 5]["ctx"])        # noqa: SLF001
                    torch.cuda.nvtx.range_push(f"{v}_vision")
                    sp[v].gv.replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push(f"{v}_vlm")
                    sp[v].gl.replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    print(f"  captured {list(hs)}; nonfinite {fin.read()}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("gate", "equiv", "verdict", "time", "ncu"), required=True)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--n-inputs", type=int, default=64, help="smoke only: fewer than the 64 fixed inputs")
    a = ap.parse_args()
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"] and a.mode != "verdict":
        raise SystemExit("MAXN only (T017)")
    if a.mode == "ncu":
        return mode_ncu(a)
    out = REPO / a.outdir
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if a.mode == "verdict":
        summ = mode_verdict(a, out)
        me = json.loads((out / "meta_equiv.json").read_text())
        mg = json.loads((out / "meta_gate.json").read_text())
        keep = ("bench_commit", "model_commit", "checkpoint", "checkpoint_commit", "lerobot_commit", "rung",
                "rung_note", "N", "power_mode", "sm_count", "granted_partitions", "gpu_clock_hz", "emc_rate_hz",
                "throttled", "inputs", "noise_seeds", "compilation", "implementation", "nocam_note")
        C.write_meta(out / "meta.json", {
            "experiment": "T017 step 1 + 3: fp32 gate, compiled bf16 action difference, floor F, controls, verdicts",
            **{k: me[k] for k in keep if k in me},
            "variant": "pad / nocam / trimprompt / nopad (fp32 stock gate; bf16_both compiled)",
            "gate_run": {k: mg[k] for k in ("date_utc", "command", "result") if k in mg},
            "equiv_run": {k: me[k] for k in ("date_utc", "command", "result") if k in me},
            "verdicts": summ["verdicts"], "command": " ".join([sys.executable, *sys.argv]),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return 0
    if a.mode == "gate":
        res, spec, note, manifest = mode_gate(a, out)
        exp, rung, variant = "T017 step 1: fp32 equivalence gate", "stock (shipped, eager) and its fp32 cast", \
            "pad vs nocam / trimprompt / nopad (stock eager)"
    elif a.mode == "equiv":
        res, spec, note, manifest = mode_equiv(a, out)
        exp, rung, variant = "T017 step 3: compiled bf16 actions per variant", "bf16_both", \
            "bf16_both (rung, E7) + pad / nocam / trimprompt / nopad; pad_rep = same-process repeat"
    else:
        res, spec, note, manifest, lab = mode_time(a, out)
        exp, rung, variant = "T017 step 2a/2c: stages per variant, bracketed", "bf16_both", \
            f"{lab} + pad / nocam / trimprompt / nopad (+split_vision)"
    thr = C.throttle_record(g0)
    C.write_meta(out / f"meta_{a.mode}.json" if a.mode in ("gate", "equiv") else out / "meta.json", {
        "experiment": exp, "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT, "lerobot_commit": C.LEROBOT_COMMIT
        if hasattr(C, "LEROBOT_COMMIT") else "f0d2b37bebddf7e8852cb624712da0ad564601b6",
        "rung": rung, "rung_note": note, "variant": variant, "N": N,
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"], "granted_partitions": "none (whole GPU)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr,
        "inputs": str(INPUTS.relative_to(REPO)) if a.mode != "time" else "bench/fixtures/e12 (fixture length only)",
        "noise_seeds": f"{SEED0} + input" if a.mode != "time" else "0 + frame",
        "compilation": "automatic_dynamic_shapes=False (static; one compile per distinct prefix length)"
        if a.mode != "gate" else "none (eager)",
        "implementation": "bench/pi0_padtrim.py (harness-level; LeRobot untouched; shared with T016 step 3 nopad)",
        "nocam_note": "LeRobot's _preprocess_images builds the -1 dummy for the empty camera outside every timed "
                      "region; it is removed from the list before the image encoder -- no SigLIP / projector "
                      "call, no tokens, no mask entry",
        "inputs_manifest": manifest, "result": res, "command": " ".join([sys.executable, *sys.argv]),
        "torch": torch.__version__, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if a.mode == "gate" and res["gate"] != "PASS":
        return 5
    return 0 if res.get("nonfinite", 0) == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
