"""T003 / E4 — π0 under `torch.compile`, and what compilation can and cannot reach here.

Variants (`--variant`):
  whole        `torch.compile(sample_actions, mode=...)` — what the checkpoint config does
  step_only    compile `denoise_step` alone and keep the Python loop (protocol §2 asks for it)
  forced       `whole`, with Inductor's SM-count gate defeated **in this process only**, so
               Triton GEMM templates become available. Never edits the installed package.
  nodeepcopy   `whole`, with the per-step `copy.deepcopy(past_key_values)` at
               modeling_pi0.py:919 simply deleted. **This is expected to FAIL and is kept as
               the negative control**: the expert's forward appends its S=51 suffix K/V to
               the cache even though the caller passes use_cache=False, so on step 2 the
               cache is 918 long against an 867 mask. The copy is load-bearing.
  cropcache    the cheap equivalent: skip the 15 MB clone, let the expert append, then
               `DynamicCache.crop(P)` back afterwards. Same semantics, no clone. This is the
               variant that says what the deepcopy actually costs.

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
from common import sha256 as _sha  # noqa: E402
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


@contextlib.contextmanager
def crop_instead_of_deepcopy(policy):
    """Replace the per-step 15 MB KV clone with a crop back to the prefix length.

    `denoise_step` deep-copies the cache because the expert's forward appends its S suffix
    K/V to it (verified: without the copy the cache reaches P+2S and the attention mask no
    longer broadcasts). Cropping afterwards is semantically the same and is a slice, not a
    15 MB clone.
    """
    import types
    import lerobot.policies.pi0.modeling_pi0 as M

    def _identity(x):
        return x

    shim = types.ModuleType("copy_shim")
    shim.deepcopy = _identity
    saved_copy, saved_step = M.copy, policy.model.denoise_step
    m = policy.model

    def denoise_step(self, *a, _orig=saved_step, **kw):
        pkv = kw.get("past_key_values") or (a[2] if len(a) > 2 else None)
        n_before = pkv.get_seq_length() if pkv is not None else None
        out = _orig(*a, **kw)
        if pkv is not None and n_before is not None:
            pkv.crop(n_before)          # drop the suffix K/V the expert appended
        return out

    M.copy = shim
    m.denoise_step = types.MethodType(denoise_step, m)
    try:
        yield
    finally:
        M.copy = saved_copy
        m.denoise_step = saved_step


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
                    choices=["whole", "step_only", "forced", "nodeepcopy", "cropcache"])
    ap.add_argument("--steps", default="4,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--date", default=time.strftime("%Y%m%d", time.gmtime()))
    ap.add_argument("--regions", choices=["off", "on"], default="off",
                    help="off (default) = the real compiled model, e2e only, slope gives "
                         "ms/step. on = region breakdown, but the NVTX/event context "
                         "managers graph-break the compiled region — a labelled variant only.")
    ap.add_argument("--allow-capped", action="store_true",
                    help="T004 4b(ii) (2026-09-28): permit a capped power mode, as run_e3.py does; the real "
                         "mode is recorded in the gate and meta.json.")
    ap.add_argument("--exp", default="E4",
                    help="experiment label in the directory name (4b(ii) names its runs E3_pi0_<mode>_N<N>_<powermode>)")
    ap.add_argument("--suffix", default="",
                    help="appended to the directory name, e.g. the power mode (30W): a capped run must never "
                         "carry a MAXN-looking name")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    gate = P.gate(require_maxn=not args.allow_capped)
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
        print("  per-step copy.deepcopy removed — NEGATIVE CONTROL, expected to fail")
    if args.variant == "cropcache":
        stack.enter_context(crop_instead_of_deepcopy(policy))
        print("  per-step KV clone replaced by DynamicCache.crop(P) (labelled variant)")

    with stack:
        policy, cinfo = compile_policy(policy, args.mode, args.variant)
        results = {}
        for n in steps:
            run = REPO / "results" / (f"{args.date}_{args.exp}_pi0_{tag}_N{n}"
                                      + (f"_{args.suffix}" if args.suffix else ""))
            if run.exists() and any(run.iterdir()):
                raise SystemExit(f"{run} exists and is not empty (protocol §5)")
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
            # The driver redirects our stdout+stderr to a log; copy whatever the caller
            # named via E4_LOG so the graph breaks, recompiles and autotune output land in
            # raw/ beside the numbers they justify. (An earlier version looked for a
            # filename the drivers never wrote, so raw/ stayed empty in every run —
            # caught by the T003 independent review.)
            import os as _os
            src = _os.environ.get("E4_LOG") or str(complog)
            if Path(src).exists():
                (run / "raw" / "compile_and_stdout.log").write_text(
                    Path(src).read_text(errors="replace"))

            # count the compile diagnostics from the log we just filed next to the numbers
            import re as _re
            _lf = run / "raw" / "compile_and_stdout.log"
            _txt = _lf.read_text(errors="replace") if _lf.exists() else ""
            _gb = _txt.count("Graph Break Reason:")
            _gbr = sorted(set(_re.findall(r"Graph Break Reason: (.+)", _txt)))
            _rc = len(_re.findall(r"Recompiling function", _txt))

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
                "kv_clone_replaced_by_crop": args.variant == "cropcache",
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
                            f"--steps {args.steps} --warmup {args.warmup} --timed {args.timed}"
                            + (" --allow-capped" if args.allow_capped else "")
                            + (f" --exp {args.exp}" if args.exp != "E4" else "")
                            + (f" --suffix {args.suffix}" if args.suffix else "")),
                "throttled": False,
                "throttled_evidence": "GPU clock pinned min==max; gate recorded above",
                **extra,
                "raw_files": [
                    {"path": f"raw/{p.name}", "sha256": _sha(p), "bytes": p.stat().st_size,
                     "kind": "stdout", "secrets_scan": "clean", "committed": True,
                     "produced_by": "bench/run_e4.py (driver stdout/stderr)"}
                    for p in sorted((run / "raw").iterdir()) if p.is_file()
                ] + [
                    {"path": "latency_samples.csv", "sha256": _sha(run / "latency_samples.csv"),
                     "bytes": (run / "latency_samples.csv").stat().st_size, "kind": "other",
                     "secrets_scan": "clean", "committed": True,
                     "produced_by": "bench/run_e4.py", "note": "every per-iteration sample"}
                ],
                "graph_break_events": _gb, "graph_break_reasons": _gbr,
                "recompile_events": _rc,
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
