#!/usr/bin/env python3
"""Print the π0 per-step ladder and the phase split from the committed device results.

Reads only files the on-device agent committed (results/), never edits them. Run from anywhere:
    python analysis/ladder.py [--root <repo root>]
Sources: results/20260921_E3_pi0_eager_N*/{latency_summary,floor}.csv (T002),
         results/20260921_E4_pi0_summary.csv (T003),
         results/regions_E4_pi0_reduce-overhead+step_only_N*/latency_summary.csv (T003 §4.1).
"""
import argparse
import csv
import os

RESEARCH = {  # what the research side used before any measurement (docs/00_context.md, 2026-09-21)
    "floor_ms": 3.15, "bound_ms": 6.30, "eager_ms": 136.0, "compiled_ms": 90.0,
    "e2e_N4": 920.0, "split_N4": (110.0, 258.0, 543.0), "ae_share_N4": 0.591, "ae_share_N10": 0.783,
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def region_median(run, region):
    for r in read_csv(os.path.join(run, "latency_summary.csv")):
        if r["region"] == region:
            return float(r["median_ms"]), int(r["n"])
    raise KeyError(f"{region} not in {run}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    R = os.path.join(a.root, "results")

    floors = {r["peak_source"]: r for r in read_csv(os.path.join(R, "20260921_E3_pi0_eager_N4", "floor.csv"))
              if r["region"] == "ae_step"}
    f_spec, f_meas = float(floors["spec"]["floor_ms"]), float(floors["measured"]["floor_ms"])
    e4 = {r["configuration"]: r for r in read_csv(os.path.join(R, "20260921_E4_pi0_summary.csv"))}

    print("π0 (LeRobot port, lerobot/pi0_libero_finetuned_v044), AGX Orin MAXN, bf16, batch 1 — ms per denoising step")
    print(f"{'rung':46s} {'ms/step':>8s} {'x spec floor':>13s} {'x meas floor':>13s}")
    rows = [("analytic floor, spec 204.8 GB/s", f_spec), ("analytic floor, measured 187.47 GB/s", f_meas),
            ("research 'optimistic bound' (2 x 3.15)", RESEARCH["bound_ms"])]
    for cfg, label in [("eager (T002)", "measured eager, slope over N"),
                       ("default", "compiled: default"),
                       ("max-autotune-no-cudagraphs", "compiled: max-autotune-no-cudagraphs"),
                       ("max-autotune", "compiled: max-autotune"),
                       ("reduce-overhead", "compiled: reduce-overhead (best stock)"),
                       ("reduce-overhead+step_only", "compiled: step_only variant"),
                       ("max-autotune+forced", "compiled: max-autotune + is_big_gpu defeated"),
                       ("reduce-overhead+cropcache", "compiled + crop-not-clone KV (modified)")]:
        rows.append((label, float(e4[cfg]["ms_per_step_slope"])))
    rows += [("research assumption: eager (paper 543/4)", RESEARCH["eager_ms"]),
             ("research assumption: compiled (136/1.51)", RESEARCH["compiled_ms"])]
    for label, v in rows:
        print(f"{label:46s} {v:8.2f} {v / f_spec:13.2f} {v / f_meas:13.2f}")

    print("\nPhase split, eager, CUDA-event medians (T002), n per point:")
    print(f"{'N':>3s} {'vision':>8s} {'vlm_prefix':>11s} {'ae_total':>9s} {'e2e':>8s} {'AE share':>9s} {'n':>4s}")
    for N in (1, 2, 4, 6, 8, 10):
        run = os.path.join(R, f"20260921_E3_pi0_eager_N{N}")
        v, n = region_median(run, "vision")
        p, _ = region_median(run, "vlm_prefix")
        ae, _ = region_median(run, "ae_total")
        e2e, _ = region_median(run, "e2e_wall")
        print(f"{N:3d} {v:8.1f} {p:11.1f} {ae:9.1f} {e2e:8.1f} {ae / e2e:9.1%} {n:4d}")
    print(f"research (paper, N=4): vision/vlm/ae = {RESEARCH['split_N4']}, e2e {RESEARCH['e2e_N4']}, "
          f"AE share {RESEARCH['ae_share_N4']:.1%} (N=4) / {RESEARCH['ae_share_N10']:.1%} (N=10)")

    print("\nPhase split under reduce-overhead + step_only (T003 §4.1, regions instrumented):")
    for N in (4, 10):
        run = os.path.join(R, f"regions_E4_pi0_reduce-overhead+step_only_N{N}")
        v, n = region_median(run, "vision")
        p, _ = region_median(run, "vlm_prefix")
        ae, _ = region_median(run, "ae_total")
        e2e, _ = region_median(run, "e2e_wall")
        print(f"{N:3d} {v:8.1f} {p:11.1f} {ae:9.1f} {e2e:8.1f} {ae / e2e:9.1%} {n:4d}")


if __name__ == "__main__":
    main()
