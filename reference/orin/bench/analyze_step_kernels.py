"""T005 — what kinds of kernel a denoising step launches, measured, by slope.

`count_step_kernels.py` counts launches for one inference. Vision and the VLM prefix run once
per inference and the expert runs once per denoising step, so a single count cannot say what a
*step* costs. Running at two values of N and taking the slope separates them, exactly as the
campaign does for latency:

    per_step = (count(N=10) - count(N=2)) / 8

This replaces an earlier per-step budget that multiplied standalone microbenchmarks by assumed
launch counts — the multipliers were guesses and the microbenchmarks carried the harness's own
fixed timing cost. These numbers are counts from the profiler, nothing else.

    python bench/analyze_step_kernels.py --lo results/<run>_N2 --hi results/<run>_N10 --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, collect_raw, sh_out, software_versions, write_json  # noqa: E402

CATEGORY = [
    (re.compile(r"gemm|gemv|s16816|sgemm|cutlass", re.I), "GEMM (tensor/CUDA core)"),
    (re.compile(r"reduce_kernel|rsqrt|rms_norm|layer_norm|_norm_", re.I),
     "reduction / normalisation"),
    (re.compile(r"CatArrayBatchedCopy|cat_", re.I), "concatenate (KV cache)"),
    (re.compile(r"Memset", re.I), "memset"),
    # `Memcpy DtoD` and `Memcpy HtoD` are copies; without this they fell through to "other",
    # which under-counted copies by 38 launches per step.
    (re.compile(r"copy_kern|direct_copy|_copy_|Memcpy", re.I), "copy"),
    (re.compile(r"softmax", re.I), "softmax"),
    (re.compile(r"elementwise|BinaryFunctor|CUDAFunctor|gpu_kernel", re.I), "elementwise"),
]


def categorise(name: str) -> str:
    for rx, label in CATEGORY:
        if rx.search(name):
            return label
    return "other"


def counts(d: Path) -> dict[str, int]:
    f = d / "kernels_per_step.csv"
    return {r["kernel"]: int(r["launches"]) for r in csv.DictReader(f.open())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lo_d, hi_d = Path(args.lo), Path(args.hi)
    n_lo = json.loads((lo_d / "meta.json").read_text())["num_inference_steps"]
    n_hi = json.loads((hi_d / "meta.json").read_text())["num_inference_steps"]
    lo, hi = counts(lo_d), counts(hi_d)

    rows = []
    for name in sorted(set(lo) | set(hi)):
        a, b = lo.get(name, 0), hi.get(name, 0)
        per_step = (b - a) / (n_hi - n_lo)
        fixed = a - n_lo * per_step
        rows.append({"kernel": name, "category": categorise(name),
                     f"launches_N{n_lo}": a, f"launches_N{n_hi}": b,
                     "per_denoising_step": round(per_step, 2),
                     "fixed_per_inference": round(fixed, 2)})
    rows.sort(key=lambda r: -r["per_denoising_step"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "kernels_per_step_by_name.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cat = collections.Counter()
    for r in rows:
        cat[r["category"]] += r["per_denoising_step"]
    total = sum(cat.values())
    with (out / "kernels_per_step_by_category.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "launches_per_denoising_step", "pct_of_step_launches"])
        for c, v in cat.most_common():
            w.writerow([c, round(v, 2), round(100 * v / total, 1)])

    print(f"  slope over N={n_lo} -> N={n_hi}: "
          f"**{total:.0f} kernel launches per denoising step**")
    print(f"  fixed per inference (vision + VLM prefix + post): "
          f"{sum(r['fixed_per_inference'] for r in rows):.0f}\n")
    print(f"  {'category':32s} {'per step':>9s} {'share':>7s}")
    for c, v in cat.most_common():
        print(f"  {c:32s} {v:9.1f} {100*v/total:6.1f} %")
    print(f"\n  top 12 individual kernels per step:")
    for r in rows[:12]:
        print(f"    {r['per_denoising_step']:7.1f}  {r['category']:28s} {r['kernel'][:58]}")
    lo_meta = json.loads((lo_d / "meta.json").read_text())
    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E2-aux (kernels per denoising step, by slope)", "task": "T005",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "derived_from": [str(lo_d), str(hi_d)],
        "method": f"slope over N={n_lo} -> N={n_hi} of per-kernel-name launch counts from "
                  f"torch.profiler; separates the expert (once per denoising step) from vision "
                  f"and the VLM prefix (once per inference).",
        "kernels_per_denoising_step": round(total, 1),
        "fixed_launches_per_inference": round(
            sum(r["fixed_per_inference"] for r in rows), 1),
        "categories": {c: round(v, 2) for c, v in cat.most_common()},
        "model": lo_meta.get("model"), "model_revision": lo_meta.get("model_revision"),
        "model_repo": lo_meta.get("model_repo"), "model_commit": lo_meta.get("model_commit"),
        "checkpoint": lo_meta.get("checkpoint"),
        "power_mode": lo_meta.get("power_mode"), "sm_count": lo_meta.get("sm_count"),
        "gpu_hz": lo_meta.get("gpu_hz"), "gate": lo_meta.get("gate"),
        "precision": "bfloat16", "compile_mode": "none (eager)", "batch": 1,
        "torch": sw.get("torch"), "torchvision": sw.get("torchvision"),
        "triton": sw.get("triton"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/analyze_step_kernels.py --lo {lo_d} --hi {hi_d} --out {out}",
        "throttled": False,
        "throttled_evidence": "counts only; no timing is taken from these runs",
        "raw_files": collect_raw(out, "derived from the two runs in `derived_from`"),
        "notes": [
            "COUNTS ONLY. torch.profiler perturbs times, so nothing here is a latency.",
            "`kernels_per_step.csv` in the source directories carries a "
            "`launches_over_N_not_per_step` column that is total/N — NOT a per-step figure, "
            "because vision and the prefix run once per inference. This directory's slope is "
            "the per-step number.",
        ],
    })
    print(f"\nwrote {out}/kernels_per_step_by_name.csv, _by_category.csv and meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
