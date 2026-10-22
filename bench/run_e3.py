"""T004 / E3: eager end-to-end latency with the protocol §2 regions, swept over N, for one model; floors (§9).

One process per model: load once, change N through the model's own field between sweep points (never by editing
loops), write ONE result directory results/<date>_E3_<model>_eager_Nsweep/:
  latency_samples.csv  N, region, iter, ms          (every sample)
  latency_summary.csv  N, region, median/mean/p5/p95/min/max ms, n, share_of_e2e
  step_fit.csv         ae_total and e2e = a + b·N per region (ms per step by slope), residuals per N
  floor.csv            per part: bytes that must be read, FLOPs, measured / spec floors, bound
  accounting.json      the accounting pass: FLOPs (FlopCounterMode), parameter bytes per region and dtype, KV bytes
  tokens.json, raw/per_N_*.json (gate, throttle verdict, peak memory per N), raw/*.stdout.log
Precision (--precision):
  eager_fp32  torch.set_float32_matmul_precision("highest"), TF32 off (cuda.matmul and cudnn), shipped weight dtypes
  shipped     whatever the model code and torch defaults set, recorded as found
Memory-path state (T003): bench/gate.memory_state_probe() before and after every N block; a block starts only in the
normal state (up to 3 min of cool-down), and both classifications are recorded.
Timing: ≥ 20 warm-up, ≥ 100 timed inferences per N; CUDA events per region read after the iteration's synchronise;
wall clock end to end. LeRobot: e2e = predict_action_chunk (as on the Orin); preprocess (the checkpoint's processor
pipeline) and postprocess (its unnormaliser) are timed in the same iteration as separate wall-clock regions, and
e2e_full = preprocess + e2e + postprocess. GR00T: e2e = Gr00tPolicy.get_action (includes its CPU processing);
model_get_action is the model call inside it; outside_model = e2e − model_get_action (derived).
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate as G  # noqa: E402
import model_registry as MR  # noqa: E402
import regions as R  # noqa: E402

LEROBOT = ("pi0", "pi05", "smolvla")


def summarize(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]  # noqa: E731
    return {"median_ms": statistics.median(v), "mean_ms": statistics.mean(v), "p5_ms": q(.05), "p95_ms": q(.95),
            "min_ms": v[0], "max_ms": v[-1], "n": len(v)}


def set_precision(mode):
    if mode == "eager_fp32":
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


# ---------------------------------------------------------------------------------------------------- model adapters
class LeRobotAdapter:
    def __init__(self, name):
        import lerobot_common as LC
        self.LC, self.spec = LC, MR.get(name)
        self.policy, self.shipped = LC.load_policy(self.spec)
        self.weights = LC.assert_real_weights(self.policy, self.spec)
        self.batch, self.pre, self.post, self.binfo = LC.build_batch(self.policy, self.spec)
        self.obs = LC.build_obs(self.policy)
        self.tokens = LC.token_counts(self.policy, self.spec, self.batch)
        self.n_default = self.spec.n_default
        self.derive_ae_total = True
        self.provenance = {"model_repo": "https://github.com/huggingface/lerobot",
                           "model_commit": "f0d2b37bebddf7e8852cb624712da0ad564601b6",
                           "checkpoint": self.spec.checkpoint, "checkpoint_revision": self.spec.revision,
                           "shipped_config": {k: str(v) for k, v in self.shipped.items()}, "weights_check": self.weights,
                           "batch_info": self.binfo, "action_horizon": int(self.policy.config.chunk_size)}

    def set_n(self, n):
        setattr(self.policy.config, self.spec.n_field, n)
        if hasattr(self.policy.model, "config"):
            setattr(self.policy.model.config, self.spec.n_field, n)

    def instrument(self, rec):
        return R.instrument_lerobot(self.policy, rec, self.spec.container)

    def iteration(self):
        """One inference; returns wall ms of preprocess, e2e (model), postprocess."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b = self.pre(dict(self.obs))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.policy.reset()
        with torch.inference_mode():
            act = self.policy.predict_action_chunk(b)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.post(act)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return {"preprocess_wall": (t1 - t0) * 1e3, "e2e_wall": (t2 - t1) * 1e3, "postprocess_wall": (t3 - t2) * 1e3,
                "e2e_full_wall": (t3 - t0) * 1e3}, act


class GrootAdapter:
    def __init__(self, name):
        import smoke_groot as SG
        from huggingface_hub import snapshot_download
        self.SG, self.name, self.v = SG, name, MR.GROOT_VERSIONS[name]
        self.ckpt = snapshot_download(self.v["checkpoint"], revision=self.v["revision"], local_files_only=True)
        self.policy = SG.load(name, self.ckpt)
        self.weights = SG.weights_check(self.policy.model, self.ckpt)
        self.obs, self.obs_shapes = SG.build_obs(name, self.policy, self.ckpt)
        self.n_default = int(self.policy.model.action_head.num_inference_timesteps)
        self.derive_ae_total = False
        self.tokens = {}
        self.provenance = {"model_repo": "https://github.com/NVIDIA/Isaac-GR00T", "model_commit": self.v["commit"],
                           "checkpoint": self.v["checkpoint"], "checkpoint_revision": self.v["revision"],
                           "embodiment_tag": SG.TAGS[name], "obs_shapes": self.obs_shapes, "weights_check": self.weights,
                           "input_note": "seeded random pixels / state at the checkpoint's own keys and shapes "
                                         "(np.random.seed(1234), 256x256), as on the Orin (groot_phases.build_observation)"}
        if name == "groot_n17":
            self.provenance["backbone"] = "nvidia/Cosmos-Reason2-2B @ 9ce19a195e423419c349abfc86fd07178b230561"

    def set_n(self, n):
        self.policy.model.action_head.num_inference_timesteps = n

    def instrument(self, rec):
        undo, fam = R.instrument_groot(self.policy, rec)
        self.family = fam
        return undo

    def iteration(self):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            act = self.policy.get_action(self.obs)
        torch.cuda.synchronize()
        return {"e2e_wall": (time.perf_counter() - t0) * 1e3}, act

    def capture_tokens(self):
        """P and S as processed, from one get_action (forward pre-hooks on the LLM and the DiT)."""
        _, _, lm, _, _ = R.groot_parts(self.policy.model)
        dit = self.policy.model.action_head.model
        seen = {}

        def hk(name, keys):
            def f(mod, args, kwargs):
                if name in seen:
                    return
                for k in keys:
                    if torch.is_tensor(kwargs.get(k)):
                        seen[name] = list(kwargs[k].shape)
                        return
                for a in args:
                    if torch.is_tensor(a):
                        seen[name] = list(a.shape)
                        return
            return f
        hs = [lm.register_forward_pre_hook(hk("llm", ["inputs_embeds", "input_ids"]), with_kwargs=True),
              dit.register_forward_pre_hook(hk("dit_hidden", ["hidden_states"]), with_kwargs=True),
              dit.register_forward_pre_hook(hk("dit_encoder", ["encoder_hidden_states"]), with_kwargs=True)]
        self.iteration()
        for h in hs:
            h.remove()
        self.tokens = {"prefix_tokens_P": seen["llm"][1], "P_dit_encoder": seen["dit_encoder"][1],
                       "suffix_tokens_S": seen["dit_hidden"][1], "shapes": seen}


def make_adapter(name):
    return LeRobotAdapter(name) if name in LEROBOT else GrootAdapter(name)


# ---------------------------------------------------------------------------------------------------- measurement
def run_n(ad, n, warmup, timed):
    ad.set_n(n)
    rec = R.Recorder()
    undo = ad.instrument(rec)
    try:
        for _ in range(warmup):
            rec.reset()
            ad.iteration()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        oc0 = G.oc_counts()
        hz = []
        rows, counts = [], {}
        t_loop = time.perf_counter()
        for it in range(timed):
            rec.reset()
            rec.start()
            walls, _ = ad.iteration()
            rec.stop()
            r = rec.results_ms(ad.derive_ae_total)
            r.update(walls)
            if "model_get_action" in r:
                r["outside_model"] = r["e2e_wall"] - r["model_get_action"]
            for k, v in r.items():
                rows.append({"N": n, "region": k, "iter": it, "ms": v})
            for j, v in enumerate(rec.step_times_ms()):
                rows.append({"N": n, "region": f"ae_step_individual[{j}]", "iter": it, "ms": v})
            if it % 10 == 0:
                hz.append(int(open(f"{G.DEVFREQ}/cur_freq").read()))
        loop_s = time.perf_counter() - t_loop
        oc1 = G.oc_counts()
        counts = dict(rec.counts)
        dev = {"N": n, "peak_mem_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_mem_reserved_bytes": torch.cuda.max_memory_reserved(),
               "loop_wall_s": loop_s, "control_frequency_hz": timed / loop_s,
               "region_call_counts_total": counts, "throttle": G.throttle_verdict(oc0, oc1, hz),
               "gpu_hz_samples": hz}
    finally:
        undo()
    return rows, dev


def accounting_pass(ad, n):
    ad.set_n(n)
    rec = R.Recorder(account=True)
    undo = ad.instrument(rec)
    try:
        rec.reset()
        rec.start()
        ad.iteration()
        rec.stop()
    finally:
        undo()
    acc = rec.accounting()
    acc["N"] = n
    return acc


def floor_rows(model, acc, hw, n):
    bw = hw["measured_bw_GBps"]["best"]["value"]
    tf = {k: (v or {}).get("value") for k, v in hw["measured_tflops"].items() if isinstance(v, dict) and "value" in v}
    rows = []

    def peak_for(by_dtype):
        dom = max(by_dtype, key=by_dtype.get) if by_dtype else "torch.bfloat16"
        key = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}.get(dom, "bf16")
        return tf[key], key, dom

    pb, fl, bd, kv = acc["param_bytes"], acc["flops"], acc["param_bytes_by_dtype"], acc["kv_bytes"]
    kv_total = sum(kv.values())
    steps = sorted(k for k in pb if k.startswith("ae_step["))
    parts = [("vision", pb.get("vision", 0), fl.get("vision", 0), bd.get("vision", {}), "weights of the vision part (+ projector)"),
             ("vlm_prefix", pb.get("vlm_prefix", 0), fl.get("vlm_prefix", 0), bd.get("vlm_prefix", {}), "VLM / LLM weights once")]
    if steps:
        s0 = steps[0]
        parts.append(("ae_step", pb[s0] + kv_total, fl.get(s0, 0), bd.get(s0, {}),
                      f"one step ({s0}): expert weights + the hand-over it reads ({kv})"))
        uniq_total = pb.get("ae_total", pb[s0] * 1)
        parts.append(("ae_total", uniq_total + kv_total, sum(fl.get(s, 0) for s in steps) if "ae_total" not in fl else fl["ae_total"],
                      bd.get("ae_total", bd.get(s0, {})),
                      f"N={n}: bytes read at least once (expert weights once + hand-over); FLOPs of all steps"))
        parts.append(("ae_total_restream", pb[s0] * len(steps) + kv_total * len(steps),
                      sum(fl.get(s, 0) for s in steps), bd.get(s0, {}),
                      f"N={n}: expert weights re-read every step (they exceed the 32 MiB L2)"))
    for region, nbytes, flops, by_dtype, note in parts:
        peak_tf, key, dom = peak_for(by_dtype)
        for src, gbps, tfl in (("measured", bw, peak_tf), ("spec", 273.0, None)):
            fm = nbytes / (gbps * 1e9) * 1e3
            fc = flops / (tfl * 1e12) * 1e3 if tfl else None
            fmax = max(fm, fc) if fc is not None else fm
            rows.append({"model": model, "region": region, "bytes_read": nbytes, "flops": flops,
                         "dominant_param_dtype": dom, "peak_dtype": key if tfl else "n/a: no published per-dtype spec",
                         "peak_GBps": gbps, "peak_TFLOPps": tfl if tfl else "n/a",
                         "floor_mem_ms": round(fm, 4), "floor_compute_ms": round(fc, 4) if fc is not None else "n/a",
                         "floor_ms": round(fmax, 4), "bound": ("memory" if fc is None or fm >= fc else "compute"),
                         "peak_source": src if src == "spec" else "measured: results/20261022_E1_peak/hw_constants.json",
                         "note": note})
    return rows


def step_fit(summary_by_n, region):
    pts = [(n, s[region]["median_ms"]) for n, s in sorted(summary_by_n.items()) if region in s]
    if len(pts) < 3:
        return None
    k = len(pts)
    mx = sum(p[0] for p in pts) / k
    my = sum(p[1] for p in pts) / k
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    return {"region": region, "a_ms": round(a, 4), "b_ms_per_step": round(b, 4),
            "r2": round(1 - ss_res / ss_tot, 6) if ss_tot else 1.0,
            "residuals": {str(p[0]): round(p[1] - (a + b * p[0]), 4) for p in pts}, "points": len(pts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LEROBOT) + sorted(MR.GROOT_VERSIONS))
    ap.add_argument("--steps", default="1,2,4,6,8,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--precision", choices=["eager_fp32", "shipped"], default="eager_fp32")
    ap.add_argument("--hw", default="results/20261022_E1_peak/hw_constants.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--accounting-only", action="store_true",
                    help="load, run the accounting pass at the default N, write accounting.json + tokens.json, exit")
    a = ap.parse_args()
    if a.accounting_only:
        os.makedirs(a.out, exist_ok=True)
        set_precision(a.precision)
        ad = make_adapter(a.model)
        if isinstance(ad, GrootAdapter):
            ad.capture_tokens()
        acc = accounting_pass(ad, ad.n_default)
        json.dump(acc, open(os.path.join(a.out, f"accounting_{a.model}.json"), "w"), indent=1, default=str)
        json.dump(ad.tokens, open(os.path.join(a.out, f"tokens_{a.model}.json"), "w"), indent=1, default=str)
        return
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty (protocol §5)")
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    steps = [int(x) for x in a.steps.split(",")]
    hw = json.load(open(a.hw))
    t_start = time.time()
    g0 = G.wait_for_gate()
    json.dump(g0, open(os.path.join(a.out, "raw", "gate_start.json"), "w"), indent=1, default=str)
    prec_before = set_precision(a.precision)
    ad = make_adapter(a.model)
    prec_after_load = set_precision(a.precision) if a.precision == "eager_fp32" else {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if isinstance(ad, GrootAdapter):
        ad.capture_tokens()
    n_def = ad.n_default
    print(f"{a.model}: loaded; N default {n_def}; tokens {json.dumps(ad.tokens)[:300]}", flush=True)
    acc = accounting_pass(ad, n_def)
    json.dump(acc, open(os.path.join(a.out, "accounting.json"), "w"), indent=1, default=str)
    all_rows, summary_by_n, dev_by_n = [], {}, {}
    for n in steps:
        gn = G.wait_for_gate(apply_clocks=True)
        ms0 = G.memory_state_probe()
        for _ in range(6):                      # baselines in the normal memory-path state (T003 finding)
            if ms0["state"] == "normal":
                break
            time.sleep(30)
            ms0 = G.memory_state_probe()
        t0 = time.time()
        rows, dev = run_n(ad, n, a.warmup, a.timed)
        dev["memory_state_before"], dev["memory_state_after"] = ms0, G.memory_state_probe()
        dev["gate_before"] = {k: gn[k] for k in ("time_utc", "checks", "gpu_hz", "gpu_util_pct_samples", "soctherm_oc_event_cnt",
                                                 "thermal_zones_mC")}
        dev["run_s"] = time.time() - t0
        json.dump(dev, open(os.path.join(a.out, "raw", f"per_N_{n}.json"), "w"), indent=1, default=str)
        all_rows += rows
        by = {}
        for r in rows:
            by.setdefault(r["region"], []).append(r["ms"])
        summary_by_n[n] = {k: summarize(v) for k, v in by.items()}
        dev_by_n[n] = dev
        s = summary_by_n[n]
        print(f"N={n}: e2e {s['e2e_wall']['median_ms']:.1f} ms | vision {s.get('vision', {}).get('median_ms', float('nan')):.1f}"
              f" | prefix {s.get('vlm_prefix', {}).get('median_ms', float('nan')):.1f} | ae_total "
              f"{s.get('ae_total', {}).get('median_ms', float('nan')):.1f} | throttled {dev['throttle']['throttled']} | state "
              f"{dev['memory_state_before']['state']}->{dev['memory_state_after']['state']}",
              flush=True)
    with open(os.path.join(a.out, "latency_samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "region", "iter", "ms"])
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(a.out, "latency_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "region", "median_ms", "mean_ms", "p5_ms", "p95_ms", "min_ms", "max_ms", "n", "share_of_e2e"])
        for n, s in summary_by_n.items():
            e2e = s["e2e_wall"]["median_ms"]
            for k in sorted(s):
                v = s[k]
                w.writerow([n, k, round(v["median_ms"], 4), round(v["mean_ms"], 4), round(v["p5_ms"], 4),
                            round(v["p95_ms"], 4), round(v["min_ms"], 4), round(v["max_ms"], 4), v["n"],
                            round(v["median_ms"] / e2e, 4)])
    fits = [f for f in (step_fit(summary_by_n, "ae_total"), step_fit(summary_by_n, "e2e_wall")) if f]
    with open(os.path.join(a.out, "step_fit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "a_ms", "b_ms_per_step", "r2", "points", "residuals"])
        w.writeheader()
        for r in fits:
            w.writerow({**r, "residuals": json.dumps(r["residuals"])})
    fr = floor_rows(a.model, acc, hw, n_def)
    with open(os.path.join(a.out, "floor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fr[0]))
        w.writeheader()
        w.writerows(fr)
    json.dump(ad.tokens, open(os.path.join(a.out, "tokens.json"), "w"), indent=1, default=str)
    run_info = {"model": a.model, "precision_mode": a.precision, "precision_flags_before_load": prec_before,
                "precision_flags_after_load": prec_after_load, "n_default": n_def, "steps": steps,
                "warmup": a.warmup, "timed": a.timed, "provenance": ad.provenance, "tokens": ad.tokens,
                "param_dtypes": sorted({str(p.dtype) for p in (ad.policy.model.parameters())}),
                "wall_s": time.time() - t_start, "throttled_by_N": {n: d["throttle"] for n, d in dev_by_n.items()},
                "memory_state_by_N": {n: (d["memory_state_before"], d["memory_state_after"]) for n, d in dev_by_n.items()}}
    json.dump(run_info, open(os.path.join(a.out, "raw", "run_info.json"), "w"), indent=1, default=str)
    print("done", a.out, flush=True)


if __name__ == "__main__":
    main()
