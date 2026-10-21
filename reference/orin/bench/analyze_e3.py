"""T002 / E3 — turn the six result directories into the report's tables.

Produces, on stdout and as CSVs in the newest result directory's parent:
  * the per-region table at each N (median, p5-p95, n, share of end-to-end);
  * ms per denoising step by BOTH definitions protocol §2 asks for —
      (a) instrumented: the CUDA-event time of each `ae_step[i]`,
      (b) slope: least-squares fit of ae_total = a + b*N across the sweep, b = ms/step;
  * whether the steps are equal;
  * the ladder: analytic floor -> measured eager, per denoising step.

    python bench/analyze_e3.py --date 20260921
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402


def read_summary(run: Path) -> dict[str, dict]:
    out = {}
    with (run / "latency_summary.csv").open() as f:
        for r in csv.DictReader(f):
            out[r["region"]] = {k: (float(v) if k != "region" and v else v)
                                for k, v in r.items()}
    return out


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--prefix", default="E3_pi0_eager")
    args = ap.parse_args()

    runs = sorted((REPO / "results").glob(f"{args.date}_{args.prefix}_N*"),
                  key=lambda p: int(p.name.rsplit("N", 1)[1]))
    if not runs:
        print("no result directories found", file=sys.stderr)
        return 1
    data = {}
    for run in runs:
        n = int(run.name.rsplit("N", 1)[1])
        data[n] = {"summary": read_summary(run),
                   "meta": json.loads((run / "meta.json").read_text()),
                   "path": run}
    Ns = sorted(data)

    # ---------------------------------------------------------------- regions
    print("\n=== per-region median ms (n = timed iterations) ===")
    regions = ["preprocess", "vision", "vlm_prefix", "ae_total", "e2e_wall"]
    hdr = f"{'region':12s}" + "".join(f"{('N='+str(n)):>12s}" for n in Ns)
    print(hdr)
    for reg in regions:
        row = f"{reg:12s}"
        for n in Ns:
            s = data[n]["summary"].get(reg)
            row += f"{s['median_ms']:12.2f}" if s else f"{'-':>12s}"
        print(row)

    print("\n=== share of end-to-end at each N (%) ===")
    print(hdr)
    for reg in ["vision", "vlm_prefix", "ae_total"]:
        row = f"{reg:12s}"
        for n in Ns:
            s = data[n]["summary"].get(reg)
            e = data[n]["summary"]["e2e_wall"]["median_ms"]
            row += f"{100*s['median_ms']/e:12.1f}" if s else f"{'-':>12s}"
        print(row)

    # ---------------------------------------------------------------- per step
    print("\n=== ms per denoising step ===")
    print(f"{'N':>4s} {'ae_total':>10s} {'(a) instr.':>12s} {'ae_total/N':>12s} "
          f"{'steps equal?':>34s}")
    per_step_rows = []
    for n in Ns:
        s = data[n]["summary"]
        ae = s["ae_total"]["median_ms"]
        indiv = [(k, v) for k, v in s.items() if k.startswith("ae_step_individual[")]
        indiv.sort(key=lambda kv: int(kv[0].split("[")[1][:-1]))
        meds = [v["median_ms"] for _, v in indiv]
        instr = sum(meds) / len(meds) if meds else float("nan")
        if meds:
            spread = f"first {meds[0]:.2f}, rest {min(meds[1:] or meds):.2f}-{max(meds[1:] or meds):.2f}"
        else:
            spread = "-"
        print(f"{n:>4d} {ae:10.2f} {instr:12.2f} {ae/n:12.2f} {spread:>34s}")
        per_step_rows.append({"N": n, "ae_total_median_ms": round(ae, 3),
                              "instrumented_mean_step_ms": round(instr, 3),
                              "ae_total_over_N_ms": round(ae / n, 3),
                              "step_medians_ms": ";".join(f"{m:.3f}" for m in meds)})

    # ---------------------------------------------------------------- slope
    ae = [data[n]["summary"]["ae_total"]["median_ms"] for n in Ns]
    e2e = [data[n]["summary"]["e2e_wall"]["median_ms"] for n in Ns]
    a_ae, b_ae, r2_ae = linfit([float(n) for n in Ns], ae)
    a_e2, b_e2, r2_e2 = linfit([float(n) for n in Ns], e2e)
    print(f"\n=== (b) slope fit across N = {Ns} ===")
    print(f"  ae_total  = {a_ae:8.2f} + {b_ae:7.3f} * N   (r2 = {r2_ae:.5f})  "
          f"-> {b_ae:.3f} ms per denoising step")
    print(f"  e2e_wall  = {a_e2:8.2f} + {b_e2:7.3f} * N   (r2 = {r2_e2:.5f})")

    # ---------------------------------------------------------------- ladder
    floor = None
    with (data[Ns[0]]["path"] / "floor.csv").open() as f:
        for r in csv.DictReader(f):
            if r["region"] == "ae_step" and r["peak_source"] == "spec":
                floor = float(r["floor_ms"])
                floor_bytes = int(r["bytes_read"])
    print("\n=== the ladder, ms per denoising step ===")
    print(f"  analytic floor (spec 204.8 GB/s, {floor_bytes/1e6:.1f} MB)   {floor:8.2f} ms")
    print(f"  measured eager, slope                          {b_ae:8.2f} ms   "
          f"({b_ae/floor:.1f}x floor)")
    for n in (4, 10):
        if n in data:
            v = data[n]["summary"]["ae_total"]["median_ms"] / n
            print(f"  measured eager, ae_total/N at N={n:<2d}            {v:8.2f} ms   "
                  f"({v/floor:.1f}x floor)")

    out = REPO / "results" / f"{args.date}_{args.prefix}_N{Ns[-1]}" / "per_step.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_step_rows[0].keys()))
        w.writeheader()
        w.writerows(per_step_rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
