"""T004 / E4 (first bullet): π0 under PyTorch's four stock torch.compile modes, expert alone and whole policy.

One process per (mode, target) (--one), so compiled artefacts and memory never leak between configurations; the
driver (--all) runs them in sequence into one result directory, each under <out>/<mode>__<target>/.
  targets:  expert = policy.model.denoise_step   (the denoising step; the Python loop stays)
            whole  = policy.model.sample_actions (LeRobot's own compile point: modeling_pi0.py:591-593)
  modes:    default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
Precision: matmul_precision "highest", TF32 off (as eager_fp32), stated explicitly because LeRobot's own compile path
sets "high" (the Orin's TF32 confound, reference/orin/reports/E4_PARTS_ABC_GUIDE.md §4).
Per configuration: compile time (first call wall), graph breaks and recompiles (torch._dynamo counters + TORCH_LOGS
graph_breaks,recompiles,perf_hints into raw/), Inductor's "Not enough SMs" message, whether CUDA graphs were used,
peak memory, steady-state e2e latency at N ∈ --steps (≥ 20 warm-up after the last recompile, ≥ 100 timed; wall clock
around predict_action_chunk, NO region instrumentation: region wrappers inside a compiled region become graph breaks),
ms per step by slope, and the max relative difference of the actions vs eager at the same seed.
The memory-path state (bench/gate.memory_state_probe) is classified before and after each N block.

python bench/run_e4.py --all --out results/<date>_E4_pi0_compile_modes
"""
import argparse
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time

MODES = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
TARGETS = ["expert", "whole"]
HERE = os.path.dirname(os.path.abspath(__file__))


def one(mode, target, steps, warmup, timed, out):
    import torch
    sys.path.insert(0, HERE)
    import gate as G
    import lerobot_common as LC
    import model_registry as MR

    os.makedirs(out, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logs = []

    class Cap(logging.Handler):
        def emit(self, r):
            logs.append(f"{r.name}: {r.getMessage()}")
    cap = Cap()
    for name in ("torch._inductor", "torch._dynamo", "torch._inductor.utils", "torch._inductor.cudagraph_trees"):
        logging.getLogger(name).addHandler(cap)
    g0 = G.wait_for_gate()
    spec = MR.get("pi0")
    policy, shipped = LC.load_policy(spec)
    batch, _, _, _ = LC.build_batch(policy, spec)
    m = policy.model

    def set_n(n):
        policy.config.num_inference_steps = n
        m.config.num_inference_steps = n

    def call(seed=0):
        policy.reset()
        torch.manual_seed(seed)
        with torch.inference_mode():
            return policy.predict_action_chunk(batch)

    set_n(steps[-1])
    ref = call().float().clone()
    from torch._dynamo.utils import counters
    counters.clear()
    t0 = time.perf_counter()
    if target == "expert":
        m.denoise_step = torch.compile(m.denoise_step, mode=mode)
    else:
        m.sample_actions = torch.compile(m.sample_actions, mode=mode)
    act = call()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    rel = ((act.float() - ref).norm() / ref.norm()).item()
    res = {"mode": mode, "target": target, "compile_first_call_s": compile_s, "rel_l2_vs_eager_same_seed": rel,
           "precision": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                         "allow_tf32": torch.backends.cuda.matmul.allow_tf32},
           "gate_start": {k: g0[k] for k in ("time_utc", "checks", "gpu_hz", "soctherm_oc_event_cnt")}, "by_N": {}}
    for n in steps:
        set_n(n)
        rc_before = sum(counters["recompiles"].values()) if "recompiles" in counters else 0
        tw = time.perf_counter()
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - tw
        ms0 = G.memory_state_probe()
        for _ in range(6):
            if ms0["state"] == "normal":
                break
            time.sleep(30)
            ms0 = G.memory_state_probe()
        torch.cuda.reset_peak_memory_stats()
        oc0 = G.oc_counts()
        walls = []
        for _ in range(timed):
            torch.cuda.synchronize()
            t = time.perf_counter()
            call()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t) * 1e3)
        oc1 = G.oc_counts()
        v = sorted(walls)
        res["by_N"][n] = {"median_ms": statistics.median(v), "mean_ms": statistics.mean(v),
                          "p5_ms": v[int(round(.05 * (len(v) - 1)))], "p95_ms": v[int(round(.95 * (len(v) - 1)))],
                          "min_ms": v[0], "n": len(v), "warmup_s": warm_s,
                          "recompiles_during_warmup": (sum(counters["recompiles"].values()) if "recompiles" in counters else 0) - rc_before,
                          "peak_mem_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "throttle": G.throttle_verdict(oc0, oc1), "memory_state_before": ms0,
                          "memory_state_after": G.memory_state_probe(), "samples_ms": walls}
        print(f"{mode} {target} N={n}: {res['by_N'][n]['median_ms']:.1f} ms", flush=True)
    pts = [(n, res["by_N"][n]["median_ms"]) for n in steps]
    k = len(pts)
    mx, my = sum(p[0] for p in pts) / k, sum(p[1] for p in pts) / k
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    res["slope_ms_per_step"], res["intercept_ms"] = b, my - b * mx
    res["dynamo_counters"] = {c: dict(v) for c, v in counters.items()}
    res["graph_breaks"] = sum(counters["graph_break"].values()) if "graph_break" in counters else 0
    res["not_enough_sms_logged"] = any("Not enough SMs" in s for s in logs)
    res["cudagraph_skips_logged"] = [s for s in logs if "skipping cudagraphs" in s.lower()][:20]
    res["cudagraphs_expected"] = mode in ("reduce-overhead", "max-autotune")
    res["log_lines"] = len(logs)
    open(os.path.join(out, "inductor_dynamo_log.txt"), "w").write("\n".join(logs))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("MODE", "TARGET"))
    ap.add_argument("--steps", default="2,6,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    if a.one:
        one(a.one[0], a.one[1], steps, a.warmup, a.timed, a.out)
        return
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    rows = []
    for target in TARGETS:
        for mode in MODES:
            sub = os.path.join(a.out, "raw", f"{mode}__{target}")
            os.makedirs(sub, exist_ok=True)
            env = dict(os.environ, TORCH_LOGS="graph_breaks,recompiles,perf_hints",
                       TORCHINDUCTOR_CACHE_DIR=os.path.join(os.environ.get("E4_CACHE_ROOT", "/tmp"), f"inductor_{mode}_{target}"),
                       TRITON_CACHE_DIR=os.path.join(os.environ.get("E4_CACHE_ROOT", "/tmp"), f"triton_{mode}_{target}"))
            with open(os.path.join(sub, "stdout.log"), "w") as lf:
                rc = subprocess.run([sys.executable, __file__, "--one", mode, target, "--steps", a.steps, "--warmup",
                                     str(a.warmup), "--timed", str(a.timed), "--out", sub], stdout=lf,
                                    stderr=subprocess.STDOUT, env=env).returncode
            r = {"mode": mode, "target": target, "exit": rc}
            # torch 2.10 never populates counters["recompiles"]; the evidence is the TORCH_LOGS=recompiles stream
            # ("[__recompiles]" lines), checked against the graph-break stream of the same log (T004 review)
            logtxt = open(os.path.join(sub, "stdout.log")).read()
            r["recompile_log_lines"] = logtxt.count("[__recompiles]")
            r["graph_break_log_lines"] = logtxt.count("[__graph_breaks]")
            rp = os.path.join(sub, "result.json")
            if rc == 0 and os.path.exists(rp):
                d = json.load(open(rp))
                r.update({"compile_first_call_s": round(d["compile_first_call_s"], 1),
                          "graph_breaks": d["graph_breaks"], "slope_ms_per_step": round(d["slope_ms_per_step"], 3),
                          "intercept_ms": round(d["intercept_ms"], 2), "not_enough_sms_logged": d["not_enough_sms_logged"],
                          "cudagraphs_expected": d["cudagraphs_expected"], "cudagraph_skips": len(d["cudagraph_skips_logged"]),
                          "rel_l2_vs_eager": f"{d['rel_l2_vs_eager_same_seed']:.2e}"})
                for n, v in d["by_N"].items():
                    r[f"N{n}_median_ms"] = round(v["median_ms"], 2)
                    r[f"N{n}_p95_ms"] = round(v["p95_ms"], 2)
                    r[f"N{n}_recompiles"] = v["recompiles_during_warmup"]
                    r[f"N{n}_peak_mem_GB"] = round(v["peak_mem_allocated_bytes"] / 1e9, 2)
                    r[f"N{n}_throttled"] = v["throttle"]["throttled"]
                    r[f"N{n}_state"] = f"{v['memory_state_before']['state']}->{v['memory_state_after']['state']}"
            rows.append(r)
            print(r, flush=True)
            time.sleep(30)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("mode", "target", "exit"), k))
    with open(os.path.join(a.out, "compile_modes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
