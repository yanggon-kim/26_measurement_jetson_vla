"""T004 — what each optimisation actually reduces, one factor at a time.

The campaign has measured these optimisations in separate runs and it became possible to
attribute a gain to the wrong cause. Two concrete cases:

* E4's `step_only` compiles `denoise_step` and leaves `config.compile_model` unset, so
  TF32 is never enabled; E11f loaded the policy with `compile_mode=`, which DOES set it
  (`modeling_pi0.py:591-595` calls `torch.set_float32_matmul_precision("high")` one line
  before `torch.compile`). The prefix then looked 1.41x faster "under compile" when the
  cause was the precision flip.
* Compiling `sample_actions` (LeRobot's own path) and compiling `denoise_step` (our
  `step_only` variant) are different interventions with different scopes, and an isolated
  action-expert loop that calls `denoise_step` directly is untouched by the former.

This runs ONE ladder in which each rung differs from the one above it by exactly one
factor, and reports every region, so the attribution is read off the table instead of
argued:

  0  eager, matmul_precision=highest                    baseline
  1  eager, matmul_precision=high                       + TF32
  2  compile denoise_step, no CUDA graphs               + compilation, expert only
  3  compile denoise_step, reduce-overhead              + CUDA graphs, expert only
  4  compile sample_actions, no CUDA graphs             + compilation, whole model
  5  compile sample_actions, reduce-overhead            + CUDA graphs, whole model
  6  compile sample_actions, max-autotune                + Triton autotuning

Rungs 0-3 carry NVTX/CUDA-event regions, so `vision`, `vlm_prefix` and `ae_total` are
separated. Rungs 4-5 cannot: a region boundary inside a compiled `sample_actions` forces a
graph break, so those rungs report end-to-end only and say so. That asymmetry is the reason
`step_only` exists at all.

    python bench/opt_ladder.py --outdir results/<date>_E4c_pi0_opt_ladder --steps 4,10
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
from common import (REPO, collect_raw, compress_and_record, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

# (key, label, precision, compile_target, compile_mode, regions_possible)
RUNGS = [
    ("eager_fp32",      "eager (matmul_precision=highest)",       "highest", None,             None,              True),
    ("eager_tf32",      "+ TF32 (matmul_precision=high)",        "high",    None,             None,              True),
    ("step_compile",    "+ compile denoise_step (no CUDA graphs)","high",    "denoise_step",   "max-autotune-no-cudagraphs", True),
    ("step_graphs",     "+ CUDA graphs on denoise_step",         "high",    "denoise_step",   "reduce-overhead", True),
    ("whole_compile",   "+ compile sample_actions (no graphs)",   "high",    "sample_actions", "max-autotune-no-cudagraphs", False),
    ("whole_graphs",    "+ CUDA graphs on sample_actions",       "high",    "sample_actions", "reduce-overhead", False),
    ("whole_autotune",  "+ Triton autotuning (max-autotune)",     "high",    "sample_actions", "max-autotune",    False),
]


def time_e2e(policy, batch, warmup: int, timed: int) -> tuple[dict, list[float]]:
    fn = lambda: policy.predict_action_chunk(batch)      # noqa: E731
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return summarize(s), s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", default="4,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--rungs", default=",".join(r[0] for r in RUNGS))
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    want = [r for r in RUNGS if r[0] in args.rungs.split(",")]
    steps = [int(x) for x in args.steps.split(",") if x]

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    rows, notes, samples = [], [], []
    verified = None

    for N in steps:
        for key, label, prec, target, mode, regions_ok in want:
            print(f"\n=== N={N}  {key}: {label}", flush=True)
            # A fresh policy per rung. torch.compile is not reversible and a compiled
            # callable left on the module would contaminate every rung below it.
            policy, cfg, shipped = P.load_policy(num_inference_steps=N)
            # Verify the weights once. assert_real_weights re-reads the checkpoint and
            # compares 400 tensors; the guard is against from_pretrained silently
            # returning random weights, which cannot change between rungs of one process.
            if verified is None:
                verified = P.assert_real_weights(policy)
                print("weights:", json.dumps(verified), flush=True)
            real = verified
            fx = P.load_fixture()
            batch, _, _ = P.build_batch(policy, fx)
            torch.set_float32_matmul_precision(prec)

            compile_s = None
            if target:
                m = policy.model
                t0 = time.time()
                setattr(m, target, torch.compile(getattr(m, target), mode=mode))
                compile_s = round(time.time() - t0, 3)

            rec = undo = None
            if regions_ok:
                rec = R.Recorder(enable_layers=False, layers_in_step=0)
                undo = R.instrument(policy, rec)

            for _ in range(args.warmup):
                policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            if rec is not None:
                rec.reset()
                rec.start()
            st, s = time_e2e(policy, batch, 0, args.timed)
            if rec is not None:
                rec.stop()
            reg = rec.results_ms() if rec is not None else {}
            if undo:
                undo()

            row = {"N": N, "rung": key, "label": label, "matmul_precision": prec,
                   "compile_target": target or "", "compile_mode": mode or "",
                   "regions": "on" if regions_ok else "off (graph break)",
                   "compile_call_s": compile_s if compile_s is not None else "",
                   "e2e_median_ms": round(st["median_ms"], 4),
                   "e2e_p5_ms": round(st["p5_ms"], 4), "e2e_p95_ms": round(st["p95_ms"], 4),
                   "n": st["n"]}
            for k in ("vision", "vlm_prefix", "ae_total"):
                row[k + "_ms"] = round(reg[k], 4) if k in reg else ""
            rows.append(row)
            samples += [{"N": N, "rung": key, "iter": i, "ms": v} for i, v in enumerate(s)]
            print(f"  e2e {st['median_ms']:8.2f} ms   vision {row['vision_ms'] or '—':>10} "
                  f"vlm_prefix {row['vlm_prefix_ms'] or '—':>10} "
                  f"ae_total {row['ae_total_ms'] or '—':>10}"
                  + (f"   (compile {compile_s} s)" if compile_s else ""), flush=True)
            if not regions_ok:
                notes.append(f"N={N} {key}: regions off — a region boundary inside a "
                             f"compiled sample_actions forces a graph break, so only "
                             f"end-to-end is measurable for this rung.")
            del policy
            torch.cuda.empty_cache()

    with (out / "opt_ladder.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # marginal effect of each rung, per N, per region
    eff = []
    for N in steps:
        rs = [r for r in rows if r["N"] == N]
        for prev, cur in zip(rs, rs[1:]):
            e = {"N": N, "step": f"{prev['rung']} -> {cur['rung']}", "adds": cur["label"]}
            for k in ("e2e_median", "vision", "vlm_prefix", "ae_total"):
                col = k + "_ms"
                a, b = prev.get(col), cur.get(col)
                e[k + "_speedup"] = round(a / b, 4) if a and b else ""
            eff.append(e)
    if eff:
        with (out / "opt_ladder_marginal.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff[0].keys()))
            w.writeheader()
            w.writerows(eff)

    sp = out / "raw" / "latency_samples.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "rung", "iter", "ms"])
        w.writeheader()
        w.writerows(samples)
    orig = compress_and_record(sp)

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E4c (optimisation ladder)", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_repo": P.CHECKPOINT, "model_commit": P.REVISION,
        "model_revision": P.REVISION, "checkpoint": P.CHECKPOINT, "precision": "bfloat16",
        "compile_mode": "per rung, see opt_ladder.csv", "batch": 1,
        "num_denoise_steps": steps, "prefix_tokens": 816, "suffix_tokens": 51,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/opt_ladder.py --outdir {args.outdir} "
                   f"--steps {args.steps} --warmup {args.warmup} --timed {args.timed}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "originals": [orig], "not_measured": notes, "raw_files": [],
        "notes": [
            "A fresh policy is loaded for every rung. torch.compile is not reversible, and "
            "a compiled callable left on the module would contaminate the rungs below it.",
            "Rungs 0-3 keep regions on; 4-5 cannot, because a region boundary inside a "
            "compiled sample_actions forces a graph break. Compare rung 4/5's e2e with "
            "rung 3's e2e, never with its per-region numbers.",
            "`step_compile`/`whole_compile` use max-autotune-no-cudagraphs to get "
            "compilation WITHOUT CUDA graphs; the next rung adds reduce-overhead, whose "
            "difference from it is the CUDA-graph contribution. T003 found the stock modes "
            "differ by 1-4 % otherwise, so the mode change is a graph switch, not a "
            "codegen change.",
            "TF32 is set explicitly on every rung rather than inherited from "
            "config.compile_model, so that the precision factor is separated from the "
            "compilation factor instead of moving with it.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({len(rows)} rungs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
