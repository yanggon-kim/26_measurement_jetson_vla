"""T002 / E3 — π0 end-to-end latency, eager, swept over the number of denoising steps.

One result directory per N (protocol §5). The model is loaded once and N is changed between
sweep points via the model's own parameter (`num_inference_steps`, protocol §2) — no loop is
edited, and reloading 7 GB six times is avoided.

    python bench/run_e3.py --steps 1,2,4,6,8,10 --warmup 20 --timed 100
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
import regions as R  # noqa: E402
from common import REPO, sh_out, software_versions, summarize, write_json  # noqa: E402

SPEC_GBPS = 204.8          # docs/00_context.md; measured peaks come from T005/E1
SPEC_TFLOPS = 42.0


def floor_rows(model_bytes: dict, tokens: dict, layers: int = 18) -> list[dict]:
    """protocol §9 — the time below which no software can go, from the LOADED model."""
    P_tok, S_tok = tokens["prefix_tokens_P"], tokens["suffix_tokens_S"]
    # KV cache read by one ae_step: 2 (K,V) x layers x kv_heads x head_dim x P x 2 B (bf16)
    kv_bytes = 2 * layers * 1 * 256 * P_tok * 2
    rows = []

    def add(region, nbytes, flops, note):
        for src, gbps, tfl in (("spec", SPEC_GBPS, SPEC_TFLOPS), ("measured", None, None)):
            if gbps is None:
                rows.append({"model": "pi0_lerobot", "region": region, "bytes_read": nbytes,
                             "flops": flops, "peak_GBps": "", "peak_TFLOPps": "",
                             "floor_mem_ms": "", "floor_compute_ms": "", "floor_ms": "",
                             "bound": "", "peak_source": "measured",
                             "note": "not measured yet — T005/E1 fills the peaks"})
                continue
            fmem = nbytes / (gbps * 1e9) * 1e3
            fcomp = flops / (tfl * 1e12) * 1e3
            rows.append({"model": "pi0_lerobot", "region": region, "bytes_read": nbytes,
                         "flops": flops, "peak_GBps": gbps, "peak_TFLOPps": tfl,
                         "floor_mem_ms": round(fmem, 4), "floor_compute_ms": round(fcomp, 4),
                         "floor_ms": round(max(fmem, fcomp), 4),
                         "bound": "memory" if fmem >= fcomp else "compute",
                         "peak_source": src, "note": note})

    vis = model_bytes["vision_tower"] + model_bytes["multi_modal_projector"]
    add("vision", vis, 0, "weights of vision_tower + projector, read once per inference; "
                          "FLOPs not counted analytically (reported as 0, see the report)")
    add("vlm_prefix", model_bytes["vlm_language_model"],
        2 * P_tok * model_bytes["vlm_language_model"] // 2,
        f"VLM weights once; FLOPs = 2 x P({P_tok}) x params (bf16 -> params = bytes/2)")
    ae = model_bytes["action_expert"]
    # FLOPs of one expert step = linear layers + attention of S queries against the
    # P + S cached keys. Omitting the attention term understates it by ~17 %.
    T = P_tok + S_tok
    lin_flops = 2 * S_tok * (ae // 2)                      # bf16: params = bytes / 2
    attn_flops = layers * 2 * (2 * 8 * S_tok * T * 256)    # QK^T and (softmax)V
    add("ae_step", ae + kv_bytes, lin_flops + attn_flops,
        f"expert weights {ae} B + KV cache {kv_bytes} B (2x{layers}x1x256xP({P_tok})x2B); "
        f"FLOPs = linear 2 x S({S_tok}) x params = {lin_flops/1e9:.2f} G, plus attention "
        f"{layers} x 2 x (2 x 8 heads x S x T({T}) x 256) = {attn_flops/1e9:.2f} G")
    return rows


def run_one(policy, batch, n_steps: int, warmup: int, timed: int,
            fine_layers: bool, layers_in_step: int = 0) -> tuple[dict, list[dict], dict]:
    policy.config.num_inference_steps = n_steps
    policy.model.config.num_inference_steps = n_steps

    rec = R.Recorder(enable_layers=fine_layers, layers_in_step=layers_in_step)
    undo = R.instrument(policy, rec)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        samples: list[dict] = []
        step_samples: list[list[float]] = []
        t_wall0 = time.perf_counter()
        for it in range(timed):
            rec.reset()
            rec.start()
            t0 = time.perf_counter()
            with torch.inference_mode():
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1e3
            rec.stop()
            r = rec.results_ms()
            r["e2e_wall"] = wall
            samples.append(r)
            step_samples.append(rec.step_times_ms())
        wall_total = time.perf_counter() - t_wall0
        peak = {"max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
    finally:
        undo()

    rows = []
    for i, s in enumerate(samples):
        for region, ms in s.items():
            rows.append({"region": region, "iter": i, "ms": ms})
    for i, sts in enumerate(step_samples):
        for j, ms in enumerate(sts):
            rows.append({"region": f"ae_step_individual[{j}]", "iter": i, "ms": ms})

    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["region"], []).append(r["ms"])
    summary = {k: summarize(v) for k, v in by.items()}
    extra = {"wall_clock_total_s": round(wall_total, 3),
             "control_frequency_hz": round(timed / wall_total, 3), **peak}
    return summary, rows, extra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="1,2,4,6,8,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--date", default=time.strftime("%Y%m%d", time.gmtime()))
    ap.add_argument("--fine-layers", dest="fine_layers", action="store_true", default=True,
                    help="also time ae_layer[j] inside one step (default: on)")
    ap.add_argument("--no-fine-layers", dest="fine_layers", action="store_false",
                    help="disable ae_layer[j] timing — the control for the step-0 offset, "
                         "since ae_layer instrumentation lands on that step alone")
    ap.add_argument("--allow-capped", action="store_true",
                    help="permit a capped power mode (MODE_50W/30W/15W). gate() defaults to "
                         "require_maxn=True and raises, which is right for the MAXN campaign "
                         "but makes T004 step 4b's capped-mode latency run impossible. The "
                         "real mode is still recorded in the gate and in meta.json, so a "
                         "capped number can never be mistaken for a MAXN one.")
    ap.add_argument("--layers-in-step", type=int, default=0,
                    help="which denoising step carries the ae_layer[j] events")
    ap.add_argument("--outdir", default="",
                    help="write the (single-N) run here instead of results/<date>_E3_pi0_eager_N<n>. "
                         "Capped runs need it: the default name carries no power mode, so a 30W/15W "
                         "run would look like a MAXN one (found 2026-09-26, the 30W queue passed it "
                         "and this script had no such flag).")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",")]

    gate = P.gate(require_maxn=not args.allow_capped)
    print("gate:", json.dumps(gate))
    policy, cfg, shipped = P.load_policy(num_inference_steps=steps[0], compile_mode=None)
    integrity = P.assert_real_weights(policy)
    print("weight integrity:", integrity["verdict"], f"({integrity['matched']}/{integrity['checked']})")
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    tokens = P.token_counts(policy, batch)
    mbytes = P.region_parameter_bytes(policy)
    print("tokens:", json.dumps(tokens))
    sw = software_versions()
    bench_commit = sh_out(f"git -C {REPO} rev-parse HEAD")

    for n in steps:
        run = (REPO / args.outdir) if args.outdir else (
            REPO / "results" / f"{args.date}_E3_pi0_eager_N{n}")
        if args.outdir and len(steps) != 1:
            raise SystemExit("--outdir takes exactly one --steps value")
        if run.exists() and any(run.iterdir()):
            raise SystemExit(f"{run} exists and is not empty (protocol §5)")
        (run / "raw").mkdir(parents=True, exist_ok=True)
        print(f"\n=== N={n}  warmup {args.warmup}, timed {args.timed} ===", flush=True)
        t0 = time.time()
        summary, rows, extra = run_one(policy, batch, n, args.warmup, args.timed,
                                       args.fine_layers, args.layers_in_step)
        print(f"  {time.time()-t0:.0f}s  "
              f"e2e {summary['e2e_wall']['median_ms']:.1f} ms  "
              f"ae_total {summary['ae_total']['median_ms']:.1f} ms  "
              f"per-step {summary['ae_total']['median_ms']/n:.2f} ms")

        with (run / "latency_samples.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["region", "iter", "ms"])
            w.writeheader()
            w.writerows(rows)
        with (run / "latency_summary.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "median_ms", "mean_ms", "p5_ms", "p95_ms", "min_ms", "max_ms", "n"])
            for k, s in sorted(summary.items()):
                w.writerow([k, s["median_ms"], s["mean_ms"], s["p5_ms"], s["p95_ms"],
                            s["min_ms"], s["max_ms"], s["n"]])
        fr = floor_rows(mbytes, tokens)
        with (run / "floor.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fr[0].keys()))
            w.writeheader()
            w.writerows(fr)

        meta = {
            "experiment": "E3", "task": "T002",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "pi0 (LeRobot implementation)",
            "model_repo": "https://github.com/huggingface/lerobot",
            "model_commit": "f0d2b37bebddf7e8852cb624712da0ad564601b6",
            "model_note": "LeRobot's port, not openpi's — see env/README.md §10 for why",
            "checkpoint": P.CHECKPOINT, "checkpoint_revision": P.REVISION,
            "checkpoint_bytes": 7011548640,
            "weight_integrity": integrity,
            "precision": cfg.dtype, "compile_mode": "none (eager)",
            "compile_mode_note": "eager; the checkpoint ships compile_model=true with "
                                 "compile_mode=max-autotune and both were overridden BEFORE "
                                 "construction (PI0Pytorch.__init__ compiles immediately)",
            "checkpoint_config_as_shipped": shipped,
            "num_denoise_steps": n, "action_horizon": cfg.chunk_size,
            "prefix_tokens": tokens["prefix_tokens_P"],
            "prefix_tokens_valid": tokens["prefix_tokens_valid"],
            "suffix_tokens": tokens["suffix_tokens_S"],
            "tokens_detail": tokens,
            "region_parameter_bytes": mbytes,
            "batch": 1, "n_warmup": args.warmup, "n_timed": args.timed,
            "seed": "not set — sample_noise draws from the global RNG each iteration. "
                    "Latency is shape-determined, so this does not affect the timings; "
                    "the model INPUT (the fixture) is fixed.",
            "ae_layer_instrumented_step": args.layers_in_step if args.fine_layers else None,
            "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
            "jetson_clocks": f"applied before the run; GPU locked "
                             f"{gate['gpu_min_freq_hz']}..{gate['gpu_max_freq_hz']} Hz, "
                             f"cur_freq {gate['gpu_cur_freq_hz']} Hz",
            "gpu_hz": gate["gpu_cur_freq_hz"],
            "emc_hz": int(sh_out("sudo -A cat /sys/kernel/debug/bpmp/debug/clk/emc/rate") or 0),
            "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
            "cuda": sw.get("torch_cuda"), "cudnn": sw.get("torch_cudnn"),
            "torch": sw.get("torch"), "torchvision": sw.get("torchvision"),
            "triton": sw.get("triton"), "numpy": sw.get("numpy"),
            "transformers": __import__("transformers").__version__,
            "python": sw.get("python"), "venv": sw.get("executable"),
            "bench_commit": bench_commit,
            "bench_files_sha256": sw.get("bench_files_sha256"),
            "fixture": {k: fx[k] for k in
                        ("dataset", "episode_index", "frame_index", "task_index", "task")},
            "command": f"python bench/run_e3.py --steps {args.steps} "
                       f"--warmup {args.warmup} --timed {args.timed}",
            "throttled": False,
            "throttled_evidence": "GPU clock locked min==max and cur_freq equal to it before "
                                  "and after the run; see gpu_hz",
            **extra,
            "raw_files": [],
            "notes": [
                "LeRobot's denoising loop is `for step in range(num_steps)` with a Python "
                "float time (flow_matching/euler loop). openpi's is `while time >= -dt/2` on "
                "CUDA tensors, which forces a CPU-GPU sync per step — protocol §2 describes "
                "the openpi behaviour; it does NOT apply to this implementation.",
                "Both prefill and denoise force _attn_implementation='eager' "
                "(modeling_pi0.py), so no SDPA/flash path is taken.",
                "predict_action_chunk is timed, not select_action: select_action serves 49 of "
                "every 50 calls from an action queue without running the model.",
            ],
        }
        write_json(run / "meta.json", meta)
        print(f"  wrote {run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
