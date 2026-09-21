"""T003 / E4 — π0 under `torch.compile`, and what compilation can and cannot reach here.

Variants (`--variant`):
  whole        `torch.compile(sample_actions, mode=...)` — what the checkpoint config does
  step_only    compile `denoise_step` alone and keep the Python loop (protocol §2 asks for it)
  forced       `whole`, with Inductor's SM-count gate defeated **in this process only**, so
               Triton GEMM templates become available. Never edits the installed package.
  nodeepcopy   `whole`, with the per-step `copy.deepcopy(past_key_values)` at
               modeling_pi0.py:919 hoisted out. T002 measured that line at 5.68 ms/step
               (10.4 % of a step); it is also a plausible reason CUDA graphs fail to capture.

Compile logs (`graph_breaks`, `recompiles`) go to the run's `raw/`. Timing uses the same
regions and CUDA events as T002, except that under CUDA graphs the per-region events are
unreliable, so the slope over N is the primary per-step number (protocol §2).

    python bench/run_e4.py --mode max-autotune --steps 4,10 --variant whole
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from common import REPO, sh_out, software_versions, summarize, write_json  # noqa: E402
from run_e3 import floor_rows, run_one  # noqa: E402


def run_uninstrumented(policy, batch, n_steps, warmup, timed):
    """Time the compiled model with NO region instrumentation.

    This is the primary E4 measurement. `bench/regions.py` wraps calls in Python context
    managers, and dynamo cannot inline `contextlib.__enter__`: each region boundary becomes a
    **graph break**, so an instrumented compiled run measures a graph that compilation never
    actually produces. Protocol §2 anticipates this ("events inside the compiled region cause
    graph breaks or disappear, so the slope is the primary number there"). Here we time only
    end-to-end wall clock and recover ms/step from the slope over N.
    """
    policy.config.num_inference_steps = n_steps
    policy.model.config.num_inference_steps = n_steps
    with torch.inference_mode():
        for _ in range(warmup):
            policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows, t_all0 = [], time.perf_counter()
    for it in range(timed):
        t0 = time.perf_counter()
        with torch.inference_mode():
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        rows.append({"region": "e2e_wall", "iter": it,
                     "ms": (time.perf_counter() - t0) * 1e3})
    wall = time.perf_counter() - t_all0
    summary = {"e2e_wall": summarize([r["ms"] for r in rows])}
    extra = {"wall_clock_total_s": round(wall, 3),
             "control_frequency_hz": round(timed / wall, 3),
             "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
             "max_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
    return summary, rows, extra

MODES = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


@contextlib.contextmanager
def force_gemm_templates():
    """Defeat Inductor's `is_big_gpu` SM-count gate for this process only.

    torch/_inductor/utils.py hard-codes `min_sms = 68` for CUDA; the Orin has 16, so
    max-autotune silently runs without Triton GEMM templates (T001). This patches the
    checking function in memory so the cost of that restriction can be measured. The
    installed package is never modified.
    """
    import torch._inductor.utils as U
    names = [n for n in ("is_big_gpu", "_is_big_gpu") if hasattr(U, n)]
    if not names:
        raise SystemExit("could not find Inductor's SM-count gate — report the torch version")
    saved = {n: getattr(U, n) for n in names}
    for n in names:
        setattr(U, n, lambda *a, **k: True)
    # the decision is cached per device in some versions
    for cache_attr in ("_gpu_is_big_cache",):
        if hasattr(U, cache_attr):
            getattr(U, cache_attr).clear()
    try:
        yield names
    finally:
        for n, f in saved.items():
            setattr(U, n, f)


@contextlib.contextmanager
def no_per_step_deepcopy(policy):
    """Hoist `copy.deepcopy(past_key_values)` out of denoise_step (modeling_pi0.py:919).

    The expert branch passes use_cache=False, so the callee does not mutate the cache. We
    replace copy.deepcopy with identity *while denoise_step runs* — measured as a labelled
    variant, never as the default.
    """
    import types
    import lerobot.policies.pi0.modeling_pi0 as M

    def _identity(x):
        return x

    shim = types.ModuleType("copy_shim")
    shim.deepcopy = _identity          # a plain function dynamo can inline
    saved = M.copy
    M.copy = shim
    try:
        yield
    finally:
        M.copy = saved


def compile_policy(policy, mode: str, variant: str):
    """Returns (policy, info). Compilation happens before any timing."""
    info = {"variant": variant, "mode": mode}
    m = policy.model
    t0 = time.time()
    if variant == "step_only":
        m.denoise_step = torch.compile(m.denoise_step, mode=mode)
        info["compiled"] = "PI0Pytorch.denoise_step"
    else:
        m.sample_actions = torch.compile(m.sample_actions, mode=mode)
        info["compiled"] = "PI0Pytorch.sample_actions"
    info["compile_call_s"] = round(time.time() - t0, 3)
    return policy, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--variant", default="whole",
                    choices=["whole", "step_only", "forced", "nodeepcopy"])
    ap.add_argument("--steps", default="4,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--date", default=time.strftime("%Y%m%d", time.gmtime()))
    ap.add_argument("--regions", choices=["off", "on"], default="off",
                    help="off (default) = the real compiled model, e2e only, slope gives "
                         "ms/step. on = region breakdown, but the NVTX/event context "
                         "managers graph-break the compiled region — a labelled variant only.")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    gate = P.gate()
    tag = args.mode if args.variant == "whole" else f"{args.mode}+{args.variant}"
    print(f"=== E4  mode={args.mode}  variant={args.variant}  N={steps} ===")
    print("gate:", json.dumps(gate))

    # compile logs -> the run's raw/ (TORCH_LOGS must be set before the first compile)
    logdir = REPO / "scratch" / "T003"
    logdir.mkdir(parents=True, exist_ok=True)
    complog = logdir / f"compile_{tag.replace('/','_')}_{'-'.join(map(str,steps))}.log"

    policy, cfg, shipped = P.load_policy(num_inference_steps=steps[0], compile_mode=None)
    integrity = P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    tokens = P.token_counts(policy, batch)
    mbytes = P.region_parameter_bytes(policy)

    stack = contextlib.ExitStack()
    forced_names = None
    if args.variant == "forced":
        forced_names = stack.enter_context(force_gemm_templates())
        print(f"  Inductor SM gate patched in-process: {forced_names}")
    if args.variant == "nodeepcopy":
        stack.enter_context(no_per_step_deepcopy(policy))
        print("  per-step copy.deepcopy(past_key_values) hoisted out (labelled variant)")

    with stack:
        policy, cinfo = compile_policy(policy, args.mode, args.variant)
        results = {}
        for n in steps:
            run = REPO / "results" / f"{args.date}_E4_pi0_{tag}_N{n}"
            (run / "raw").mkdir(parents=True, exist_ok=True)
            print(f"\n--- N={n} ---", flush=True)
            t0 = time.time()
            # first call includes compilation; time it separately from the steady state
            policy.config.num_inference_steps = n
            policy.model.config.num_inference_steps = n
            with torch.inference_mode():
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            first_call = time.time() - t0
            print(f"  first call (incl. compile): {first_call:.1f} s")

            if args.regions == "on":
                summary, rows, extra = run_one(policy, batch, n, args.warmup, args.timed,
                                               fine_layers=False)
            else:
                summary, rows, extra = run_uninstrumented(policy, batch, n,
                                                          args.warmup, args.timed)
            results[n] = summary
            e2e = summary["e2e_wall"]["median_ms"]
            ae = summary.get("ae_total", {}).get("median_ms")
            print(f"  e2e {e2e:.1f} ms" + (f"  ae_total {ae:.1f} ms  per-step {ae/n:.2f} ms"
                                           if ae else "  (ae_total unavailable — graph)"))

            with (run / "latency_samples.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["region", "iter", "ms"])
                w.writeheader(); w.writerows(rows)
            with (run / "latency_summary.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["region","median_ms","mean_ms","p5_ms","p95_ms","min_ms","max_ms","n"])
                for k, s in sorted(summary.items()):
                    w.writerow([k, s["median_ms"], s["mean_ms"], s["p5_ms"], s["p95_ms"],
                                s["min_ms"], s["max_ms"], s["n"]])
            fr = floor_rows(mbytes, tokens)
            with (run / "floor.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(fr[0].keys()))
                w.writeheader(); w.writerows(fr)
            if complog.exists():
                (run / "raw" / "compile.log").write_text(complog.read_text())

            sw = software_versions()
            write_json(run / "meta.json", {
                "experiment": "E4", "task": "T003",
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "pi0 (LeRobot implementation)",
                "model_repo": "https://github.com/huggingface/lerobot",
                "model_commit": "f0d2b37bebddf7e8852cb624712da0ad564601b6",
                "checkpoint": P.CHECKPOINT, "checkpoint_revision": P.REVISION,
                "checkpoint_sha256": ("31633cfa2131c058ac1a00fddec9f5731ca1"
                                      "7842af46d802cc75fea0f725a386"),
                "weight_integrity": integrity, "precision": cfg.dtype,
                "compile_mode": args.mode, "compile_variant": args.variant,
                "regions_instrumented": args.regions == "on",
                "regions_note": ("Region instrumentation is OFF: bench/regions.py uses Python "
                                 "context managers and dynamo cannot inline contextlib."
                                 "__enter__, so every region boundary becomes a graph break "
                                 "and an instrumented compiled run measures a graph "
                                 "compilation never produced. ms/step comes from the slope "
                                 "over N (protocol §2)."
                                 if args.regions == "off" else
                                 "Region instrumentation is ON: the compiled region is "
                                 "graph-broken at every region boundary. Labelled variant; "
                                 "not the headline compiled number."),
                "compile_info": cinfo,
                "compile_first_call_s": round(first_call, 3),
                "inductor_sm_gate_patched": forced_names,
                "deepcopy_hoisted": args.variant == "nodeepcopy",
                "checkpoint_config_as_shipped": shipped,
                "num_denoise_steps": n, "action_horizon": cfg.chunk_size,
                "prefix_tokens": tokens["prefix_tokens_P"],
                "suffix_tokens": tokens["suffix_tokens_S"],
                "tokens_detail": tokens, "region_parameter_bytes": mbytes,
                "batch": 1, "n_warmup": args.warmup, "n_timed": args.timed,
                "seed": "not set; latency is shape-determined and the observation is fixed",
                "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
                "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
                "jetson_clocks": "applied before the run; GPU locked min==max",
                "emc_hz": int(sh_out("sudo -n cat /sys/kernel/debug/bpmp/debug/clk/emc/rate") or 0),
                "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
                "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
                "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
                "numpy": sw.get("numpy"),
                "transformers": __import__("transformers").__version__,
                "python": sw.get("python"), "venv": sw.get("executable"),
                "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
                "bench_files_sha256": sw.get("bench_files_sha256"),
                "fixture": {k: fx[k] for k in ("dataset","episode_index","frame_index","task")},
                "command": (f"TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py "
                            f"--mode {args.mode} --variant {args.variant} "
                            f"--steps {args.steps} --warmup {args.warmup} --timed {args.timed}"),
                "throttled": False,
                "throttled_evidence": "GPU clock pinned min==max; gate recorded above",
                **extra, "raw_files": [],
                "notes": [
                    "ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on "
                    "~2.6 ms — T002 review finding).",
                    "Under CUDA graphs the per-region CUDA events may be absorbed into the "
                    "graph; where ae_total is missing or implausible the slope over N is the "
                    "per-step number (protocol §2).",
                ],
            })
            print(f"  wrote {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
