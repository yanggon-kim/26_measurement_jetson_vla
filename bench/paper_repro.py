"""T004 / Part A — reproduce arXiv:2604.24447's pi0-on-AGX-Orin measurement exactly.

Their configuration is not stated in the paper; it is INFERRED from their own published
aggregates, and the inference is recorded here so it can be challenged:

  N = 10 denoising steps.  Their Action Expert is 543.2 ms (59 % of 920.6). Divided by 10
    that is 54.32 ms/step against our measured 53.30 ms/step at N = 10 — 1.9 % apart on the
    same silicon. At N = 4 theirs would be 135.79 ms/step, 2.3x slower than ours, which is
    not credible. (Their prose says pi0 "requires only 4 denoising steps"; that sentence is
    inconsistent with their own measurement.)

  Eager — no torch.compile, no CUDA graphs.  Their Table 3 presents compilation as a
    separate optimisation applied afterwards (RTX 4090 102.3 -> 35.2 ms, Jetson Thor
    246.0 -> 163.0, Ascend 310P 818.0 -> 350.0), and Table 1's 4090 entry of 102.3 ms is
    exactly Table 3's "Baseline". So Figure 5's phase breakdown is the uncompiled baseline.

  bf16 model with the vision path NOT in plain fp32.  Their vision is 110.5 ms, matching our
    TF32 vision (103.16 ms, 1.07x) and 2.2x faster than our fp32 vision (239.60 ms).

Targets to reproduce (their Table 1 / Figure 5 / the leaderboard at
https://vla-leaderboard-01.vercel.app/):

    end-to-end       920.58 ms
    time split       vision 12 % / VLM 28 % / Action Expert 59 %
    SM utilisation   69.4 % / 97.7 % / 22.3 %      <- bench/phase_utilisation.py, needs root
    energy           1866 J                         <- their column takes only 3 distinct
                                                       values across 7 models, so it looks
                                                       assigned per model-class, not measured

This script covers latency, the phase split and energy. SM utilisation needs GPU metrics
sampling under sudo and lives in bench/phase_utilisation.py.

    python bench/paper_repro.py --outdir results/<date>_E4d_pi0_paper_config_N10
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
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, ina3221_rails,  # noqa: E402
                    sh_out, software_versions, summarize, write_json)
# tegrastats helpers live in mempressure.py; stop_tegrastats() exists because
# Popen.terminate() kills the sudo wrapper and leaves the root tegrastats running —
# T010 accumulated 22 orphans that way, growing background load under every later run.
from mempressure import start_tegrastats, stop_tegrastats  # noqa: E402

# arXiv:2604.24447, Table 1 + Figure 5, pi0 on AGX Orin
THEIRS = {"e2e_ms": 920.58, "energy_J": 1866.0,
          "time_share": {"vision": 0.12, "vlm_prefix": 0.28, "ae_total": 0.59},
          "sm_utilisation_pct": {"vision": 69.4, "vlm_prefix": 97.7, "ae_total": 22.3},
          "_source": "Table 1 / Figure 5; leaderboard https://vla-leaderboard-01.vercel.app/"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="model key from bench/model_registry.py. The N field differs "
                         "three ways across the family, so the descriptor supplies it.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--power-seconds", type=float, default=60.0)
    ap.add_argument("--tf32", choices=("on", "off"), default="on",
                    help="their vision matches ours with TF32 on; 'off' gives the plain-fp32 "
                         "control so the difference is attributable")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")

    # EAGER: compile_mode=None, so config.compile_model stays False and nothing is compiled.
    # Note that passing compile_mode= would ALSO flip TF32 on (modeling_pi0.py:591-595);
    # here TF32 is set explicitly above so the two factors stay separable.
    spec = MR.get(args.model)
    steps = args.steps if args.steps else spec.n_default
    policy, cfg, shipped = MR.load_policy(spec, n=steps, compile_mode=None)
    real = (P.assert_real_weights(policy) if args.model == "pi0"
            else {"verdict": "not verified: assert_real_weights is pi0-specific "
                              "(it diffs against pi0's own safetensors)"})
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    try:
        tok = P.token_counts(policy, batch, spec)
    except Exception as e:                                         # noqa: BLE001
        # token_counts calls embed_suffix, whose signature differs: three arguments for
        # pi0, two for pi0.5 and SmolVLA (modeling_pi05.py:684, modeling_smolvla.py:719).
        tok = {"error": f"{type(e).__name__}: {str(e)[:200]}",
               "_note": "per-model token_counts not yet written; P/S come from the run"}
    print("tokens:", json.dumps(tok), flush=True)

    rec = R.Recorder(enable_layers=False, layers_in_step=0)
    undo = R.instrument(policy, rec, spec.container)

    teg = start_tegrastats(out / "raw" / "tegrastats_paper_config.log")
    for _ in range(args.warmup):
        policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    per_iter, e2e = [], []
    for _ in range(args.timed):
        rec.reset()
        rec.start()
        t0 = time.perf_counter()
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1e3)
        rec.stop()
        per_iter.append(rec.results_ms())
    undo()

    st = summarize(e2e)
    reg = {}
    for k in {k for d in per_iter for k in d}:
        reg[k] = summarize([d[k] for d in per_iter if k in d])
    total = st["median_ms"]

    # --- energy: a sustained loop with the four INA3221 rails sampled throughout
    print(f"\npower: {args.power_seconds:.0f} s sustained loop", flush=True)
    acc, n_inf = {}, 0
    t_end = time.perf_counter() + args.power_seconds
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        policy.predict_action_chunk(batch)
        n_inf += 1
        for rail, d in ina3221_rails().items():
            acc.setdefault(rail, []).append(d.get("mW") or 0)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    stop_tegrastats(teg)

    k = min(len(v) for v in acc.values())
    module_mW = [sum(v[i] for v in acc.values()) for i in range(k)]
    p = summarize(module_mW)
    energy_J_total = p["mean_ms"] / 1000.0 * wall_s          # mean W x seconds
    energy_J_per_inf = energy_J_total / n_inf

    rows = []
    for name in ("vision", "vlm_prefix", "ae_total"):
        s = reg.get(name)
        if not s:
            continue
        rows.append({"region": name, "median_ms": round(s["median_ms"], 4),
                     "p5_ms": round(s["p5_ms"], 4), "p95_ms": round(s["p95_ms"], 4),
                     "n": s["n"], "share_of_e2e": round(s["median_ms"] / total, 4),
                     "their_ms": round(THEIRS["e2e_ms"] * THEIRS["time_share"][name], 2),
                     "their_share": THEIRS["time_share"][name],
                     "ratio_theirs_over_ours": round(
                         THEIRS["e2e_ms"] * THEIRS["time_share"][name] / s["median_ms"], 4)})
    rows.append({"region": "e2e", "median_ms": round(total, 4),
                 "p5_ms": round(st["p5_ms"], 4), "p95_ms": round(st["p95_ms"], 4),
                 "n": st["n"], "share_of_e2e": 1.0,
                 "their_ms": THEIRS["e2e_ms"], "their_share": 1.0,
                 "ratio_theirs_over_ours": round(THEIRS["e2e_ms"] / total, 4)})
    with (out / "paper_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with (out / "power.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rail", "mean_mW", "median_mW", "p5_mW",
                                          "p95_mW", "n"])
        w.writeheader()
        for rail, vals in acc.items():
            s = summarize(vals)
            w.writerow({"rail": rail, "mean_mW": round(s["mean_ms"], 1),
                        "median_mW": round(s["median_ms"], 1),
                        "p5_mW": round(s["p5_ms"], 1), "p95_mW": round(s["p95_ms"], 1),
                        "n": s["n"]})
        w.writerow({"rail": "module", "mean_mW": round(p["mean_ms"], 1),
                    "median_mW": round(p["median_ms"], 1), "p5_mW": round(p["p5_ms"], 1),
                    "p95_mW": round(p["p95_ms"], 1), "n": k})

    samp = out / "raw" / "latency_samples.csv"
    with samp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "e2e_ms"] + sorted(reg))
        for i, d in enumerate(per_iter):
            w.writerow([i, round(e2e[i], 5)] + [round(d.get(kk, float("nan")), 5)
                                                for kk in sorted(reg)])
    orig = compress_and_record(samp)

    print(f"\n{'region':12s} {'ours ms':>10s} {'share':>7s} | {'theirs ms':>10s} "
          f"{'share':>7s} {'ratio':>7s}")
    for r in rows:
        print(f"{r['region']:12s} {r['median_ms']:>10.2f} {100*r['share_of_e2e']:>6.1f}% | "
              f"{r['their_ms']:>10.2f} {100*r['their_share']:>6.1f}% "
              f"{r['ratio_theirs_over_ours']:>6.2f}x")
    print(f"\nenergy: module {p['mean_ms']/1000:.2f} W mean over {wall_s:.1f} s, "
          f"{n_inf} inferences -> {energy_J_per_inf:.2f} J/inference "
          f"({energy_J_total:.0f} J total). Theirs: {THEIRS['energy_J']} J (units unclear; "
          f"see the note in this file's docstring).")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E4d (paper configuration reproduction)", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.name, "model_repo": spec.checkpoint,
        "model_commit": MR.resolved_commit(spec),
        "model_revision": spec.revision or "not pinned (hub default branch; see model_commit)",
        "checkpoint": spec.checkpoint, "precision": "bfloat16",
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "compile_mode": "none (eager)", "batch": 1, "num_denoise_steps": steps, "n_field": spec.n_field,
        "prefix_tokens": tok.get("prefix_tokens_P"), "suffix_tokens": tok.get("suffix_tokens_S"),
        "token_counts": tok, "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/paper_repro.py --outdir {args.outdir} "
                   f"--model {args.model} --steps {steps} --tf32 {args.tf32}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "energy": {"module_mean_W": round(p["mean_ms"] / 1000, 3), "wall_s": round(wall_s, 2),
                   "inferences": n_inf, "J_per_inference": round(energy_J_per_inf, 3),
                   "J_total": round(energy_J_total, 1)},
        "paper_targets": THEIRS,
        "paper_config_inference": (
            "N=10 from their 543.2 ms Action Expert / 10 = 54.32 ms per step vs our 53.30; "
            "eager because their Table 3 treats compilation as a separate optimisation and "
            "its Baseline column equals Table 1; TF32-or-equivalent vision because their "
            "110.5 ms matches our TF32 103.16 and not our fp32 239.60."),
        "originals": [orig], "raw_files": [],
        "notes": [
            "Their energy column takes only three distinct values across seven models "
            "(1866/662/355 J), so it appears assigned per model-class rather than measured "
            "per model. Ours is four INA3221 rails sampled during a sustained loop.",
            "Regions are instrumented in eager only, which is what this configuration is; "
            "no graph break is possible because nothing is compiled.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
