"""T004 — is the "compiled" prefix speedup compilation, or the TF32 flip that comes with it?

`modeling_pi0.py:591-595` runs

    if config.compile_model:
        torch.set_float32_matmul_precision("high")      # global, enables TF32
        self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

so asking for a compiled policy also changes the numerics of every fp32 matmul in the
process, including in code `torch.compile` never touched. T010/E11f timed prefix-only and
action-expert-only loops that call `prefix_once` / `denoise_step` directly, bypassing
`sample_actions`; they therefore run eager code under both settings, yet the prefix loop
came out 1.41x faster "under reduce-overhead".

This isolates the two: one eager policy, no compilation anywhere, the SAME callables timed
with `torch.set_float32_matmul_precision` toggled and nothing else changed.

    python bench/tf32_confound.py --outdir results/<date>_E4b_pi0_tf32_confound
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
import phase_loops as PL  # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions, summarize,  # noqa: E402
                    write_json)


def bench(fn, seconds: float) -> dict:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0, s = time.perf_counter(), []
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - a) * 1e3)
    return summarize(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    policy, cfg, shipped = P.load_policy(num_inference_steps=args.steps)   # EAGER, uncompiled
    real = P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    phases = {name: PL.make_phase(policy, batch, name, args.steps)
              for name in ("prefix", "ae", "full")}

    rows, samples = [], []
    for prec in ("highest", "high"):
        torch.set_float32_matmul_precision(prec)
        for name, fn in phases.items():
            st = bench(fn, args.seconds)
            rows.append({"phase": name, "matmul_precision": prec, "compiled": False,
                         **{k: round(v, 6) if isinstance(v, float) else v
                            for k, v in st.items()}})
            print(f"  matmul_precision={prec:8s} {name:7s} median {st['median_ms']:8.2f} ms "
                  f"(p5 {st['p5_ms']:.2f}, p95 {st['p95_ms']:.2f}, n={st['n']})", flush=True)
    torch.set_float32_matmul_precision("highest")      # leave the process as we found it

    by = {(r["phase"], r["matmul_precision"]): r["median_ms"] for r in rows}
    eff = []
    for name in phases:
        a, b = by[(name, "highest")], by[(name, "high")]
        eff.append({"phase": name, "highest_ms": a, "high_ms": b,
                    "tf32_speedup": round(a / b, 4)})
        print(f"  {name:7s}: highest {a:8.2f} -> high {b:8.2f} ms   TF32 alone = {a/b:.3f}x",
              flush=True)

    for name, rs in (("tf32_confound.csv", rows), ("tf32_speedup.csv", eff)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E4b (TF32 confound)", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_repo": P.CHECKPOINT, "model_commit": P.REVISION,
        "model_revision": P.REVISION, "checkpoint": P.CHECKPOINT, "precision": "bfloat16",
        "compile_mode": "none (eager)", "batch": 1, "num_denoise_steps": args.steps,
        "prefix_tokens": 816, "suffix_tokens": 51,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": real, "shipped_config": shipped,
        "command": f"python bench/tf32_confound.py --outdir {args.outdir} "
                   f"--seconds {args.seconds} --steps {args.steps}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "raw_files": [],
        "notes": [
            "The policy is loaded WITHOUT compilation in both halves. Only "
            "torch.set_float32_matmul_precision changes between them.",
            "`prefix` and `ae` call prefix_once / denoise_step directly and never enter "
            "sample_actions, which is the callable modeling_pi0.py compiles; `full` calls "
            "predict_action_chunk and would be compiled if compilation were on. Here none "
            "of the three is compiled.",
            "Purpose: T010/E11f reported the prefix loop 1.41x faster 'under "
            "reduce-overhead' while labelling it eager_subpath. If TF32 alone reproduces "
            "that factor, the speedup is a numerics change that comes with asking for "
            "compilation, not a result of compiling anything.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
