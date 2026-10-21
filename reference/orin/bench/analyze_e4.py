"""T003 / E4 — assemble the compiled results and complete the ladder.

Reads every `results/<date>_E4_pi0_*_N*/` directory plus T002's eager runs, and produces the
mode comparison, the per-step slopes and the ladder row that replaces the research side's
~90 ms/step estimate.

    python bench/analyze_e4.py --date 20260921
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

FLOOR_MS = 3.115          # ae_step memory floor, spec 204.8 GB/s (T002 floor.csv)


def median_of(run: Path, region: str = "e2e_wall") -> float | None:
    f = run / "latency_summary.csv"
    if not f.exists():
        return None
    for r in csv.DictReader(f.open()):
        if r["region"] == region:
            return float(r["median_ms"])
    return None


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else 1.0)


def collect(date: str) -> dict[str, dict[int, float]]:
    """Collect every E4 run. UTC can roll over mid-campaign, so accept any date prefix."""
    out: dict[str, dict[int, float]] = {}
    for run in sorted((REPO / "results").glob("*_E4_pi0_*_N*")):
        m = re.match(r"\d{8}_E4_pi0_(.+)_N(\d+)$", run.name)
        if not m:
            continue
        tag, n = m.group(1), int(m.group(2))
        v = median_of(run)
        if v is not None:
            out.setdefault(tag, {})[n] = v
    for run in sorted((REPO / "results").glob("*_E3_pi0_eager_N*")):
        m = re.match(r"\d{8}_E3_pi0_eager_N(\d+)$", run.name)
        if m:
            v = median_of(run)
            if v is not None:
                out.setdefault("eager (T002)", {})[int(m.group(1))] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    data = collect(args.date)
    if not data:
        print("no E4 runs found", file=sys.stderr)
        return 1

    print(f"\n=== end-to-end median ms, and ms per denoising step (slope) ===")
    print(f"{'configuration':44s} {'N=4':>9s} {'N=10':>9s} {'ms/step':>9s} {'fixed':>9s} "
          f"{'vs eager':>9s} {'x floor':>8s}")
    eager_step = None
    rows = []
    for tag in sorted(data, key=lambda t: (t != "eager (T002)", t)):
        pts = data[tag]
        Ns = sorted(pts)
        if len(Ns) < 2:
            continue
        a, b, r2 = linfit([float(n) for n in Ns], [pts[n] for n in Ns])
        if tag == "eager (T002)":
            eager_step = b
        sp = (eager_step / b) if eager_step else float("nan")
        print(f"{tag:44s} {pts.get(4, float('nan')):9.1f} {pts.get(10, float('nan')):9.1f} "
              f"{b:9.2f} {a:9.1f} {sp:8.2f}x {b/FLOOR_MS:7.1f}x")
        rows.append({"configuration": tag, "N_points": ";".join(map(str, Ns)),
                     "e2e_N4_ms": round(pts.get(4, float('nan')), 3),
                     "e2e_N10_ms": round(pts.get(10, float('nan')), 3),
                     "ms_per_step_slope": round(b, 4), "fixed_cost_ms": round(a, 3),
                     "slope_r2": round(r2, 6),
                     "speedup_vs_eager": round(sp, 4) if eager_step else "",
                     "gap_ratio_to_floor": round(b / FLOOR_MS, 3)})

    # The ladder's "best compiled" row uses UNMODIFIED stock modes only (T003). Variants
    # that patch the model or Inductor are reported beside it, never as the ladder rung.
    MODIFIED = ("forced", "nodeepcopy", "cropcache", "step_only")
    stock = [r for r in rows if r["configuration"] != "eager (T002)"
             and not any(k in r["configuration"] for k in MODIFIED)]
    best = min(stock, key=lambda r: r["ms_per_step_slope"])
    print("\n=== the ladder, ms per denoising step (stock configurations only) ===")
    print(f"  analytic bandwidth floor                           {FLOOR_MS:8.3f} ms")
    print(f"  research-side 'optimistic bound' (2x floor)        {2*FLOOR_MS:8.3f} ms")
    print(f"  measured eager (T002)                             {eager_step:8.2f} ms  "
          f"({eager_step/FLOOR_MS:.1f}x floor)")
    print(f"  measured compiled, best stock = {best['configuration']:<18s}{best['ms_per_step_slope']:8.2f} ms  "
          f"({best['gap_ratio_to_floor']:.1f}x floor)")
    print(f"  research-side estimate for the compiled stack        ~90 ms  "
          f"({90/best['ms_per_step_slope']:.1f}x too pessimistic)")

    mods = [r for r in rows if any(k in r["configuration"] for k in MODIFIED)]
    if mods:
        print("\n=== modified variants — reported beside the ladder, not on it ===")
        for r in sorted(mods, key=lambda r: r["ms_per_step_slope"]):
            print(f"  {r['configuration']:34s} {r['ms_per_step_slope']:8.2f} ms  "
                  f"({r['gap_ratio_to_floor']:.1f}x floor, "
                  f"{r['speedup_vs_eager']:.2f}x eager)")
        print("  nodeepcopy                          FAILED — negative control: the KV deep "
              "copy is load-bearing (cache reaches P+2S, mask no longer broadcasts)")

    out = REPO / "results" / f"{args.date}_E4_pi0_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
