"""Verify a report's headline numbers against the CSVs they claim to come from.

Both adversarial review passes of T005 found the same class of defect: a number in the prose
that appears in no committed file, usually because it survived from a superseded run. This
recomputes the numbers from `results/` and checks that the report text actually contains them,
so the defect fails loudly instead of being found by a reviewer.

    python bench/check_report_numbers.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

DATE = sys.argv[1] if len(sys.argv) > 1 else "20260922"
E1 = REPO / "results" / f"{DATE}_E1_peak"
E2 = REPO / "results" / f"{DATE}_E2_ae_kernels"
KC = REPO / "results" / f"{DATE}_E2_kernel_counts"
RAILS = REPO / "results" / f"{DATE}_E7_rails_check"
REPORT = REPO / "reports" / "E1_E2_microbench.md"


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open()))


def fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    text = REPORT.read_text()
    checks: list[tuple[str, str]] = []          # (label, string that must appear)

    hw = json.loads((E1 / "hw_constants.json").read_text())
    mp = hw["measured_peaks"]
    bw = mp["peak_memory_bandwidth_GBps"]
    fl = mp["peak_bf16_TFLOPps"]
    checks += [
        ("E1 peak bandwidth", fmt(bw, 2)),
        ("E1 peak bf16", fmt(fl, 2)),
        ("E1 ridge point", fmt(mp["ridge_point_FLOP_per_byte"], 1)),
        ("eff_flops_frac", fmt(hw["efficiency_fractions"]["eff_flops_frac_measured"], 3)),
        ("eff_bw_frac", fmt(hw["efficiency_fractions"]["eff_bw_frac_measured"], 3)),
    ]

    # E2: the S=51 projection table and the 126-projection stack
    k = {(r["proj"], r["S"]): r for r in rows(E2 / "kernels.csv")}
    for p in ("gate_proj", "up_proj", "down_proj", "o_proj", "q_proj", "k_proj", "v_proj"):
        r = k[(p, "51")]
        checks.append((f"{p} x_floor @S=51", f"{float(r['x_floor']):.2f}×"))
        checks.append((f"{p} %peak @S=51", f"{float(r['pct_of_peak_bw']):.1f} %"))

    g = {r["mode"]: r for r in rows(E2 / "cuda_graph.csv")}
    wb = int(g["eager"]["weight_bytes_total"])
    floor = wb / (bw * 1e9) * 1e3
    checks += [
        ("stack weight bytes", f"{wb/1e6:.2f} MB"),
        ("stack floor", fmt(floor, 3)),
        ("stack graph ms", fmt(float(g["cuda_graph_replay"]["median_ms"]), 3)),
        ("stack eager ms", fmt(float(g["eager"]["median_ms"]), 3)),
        ("stack graph x floor",
         f"{float(g['cuda_graph_replay']['median_ms'])/floor:.2f}×"),
        ("stack eager x floor", f"{float(g['eager']['median_ms'])/floor:.2f}×"),
    ]

    lc = {r["method"]: r for r in rows(E2 / "launch_cost.csv")}
    for m in ("empty_kernel", "chain_64", "chain_64_cuda_graph"):
        checks.append((f"launch {m}", fmt(float(lc[m]["us"]), 1)))

    # the marginal/fixed fit MUST come from the committed chain totals, not from a prior run
    # exactly `chain_<k>` — `chain_64_cuda_graph` also splits to "64" and would overwrite it
    tot = {int(m.split("_")[1]): float(lc[m]["median_ms"]) * 1e3
           for m in lc if re.fullmatch(r"chain_\d+", m)}
    ks = sorted(tot)
    n = len(ks)
    mx = sum(ks) / n
    my = sum(tot[k] for k in ks) / n
    marg = (sum((k - mx) * (tot[k] - my) for k in ks)
            / sum((k - mx) ** 2 for k in ks))
    fixed = my - marg * mx
    checks += [("chain fit marginal us", fmt(marg, 2)),
               ("chain fit fixed us", fmt(fixed, 2))]

    # the allocating-clone rows: size -> GB/s mapping was printed backwards once
    cl = [r for r in rows(E1 / "latency_summary.csv") if r["kernel"] == "clone_allocating"]
    cl.sort(key=lambda r: int(r["shape"]))
    checks.append(("clone_allocating ascending GB/s",
                   " / ".join(fmt(float(r["GBps"]), 2) for r in cl)))

    # kernel counts
    kc = json.loads((KC / "meta.json").read_text())
    cat = rows(KC / "kernels_per_step_by_category.csv")
    total = sum(float(r["launches_per_denoising_step"]) for r in cat)
    checks.append(("kernels per step", f"{total:.0f}"))
    for r in cat[:3]:
        checks.append((f"category {r['category']}", f"{float(r['pct_of_step_launches']):.1f} %"))

    # ncu validation
    for r in rows(E2 / "ncu" / "sysmem_miss_validation.csv"):
        if r["kernel"] == "gemm_cold":
            checks.append(("ncu gemm_cold ratio_read", f"{float(r['ratio_read']):.4f}"))

    # rails
    pr = {(r["load"], r["rail"]): r for r in rows(RAILS / "power_rails_check.csv")}
    checks.append(("rail GPU delta on GEMM",
                   f"{float(pr[('gemm_8192_bf16', 'VDD_GPU_SOC')]['above_idle_mW']):.0f}"))

    bad = [(lab, s) for lab, s in checks if s not in text]
    for lab, s in checks:
        mark = "ok  " if s in text else "MISS"
        print(f"  {mark} {lab:34s} {s}")
    if bad:
        print(f"\n{len(bad)} value(s) recomputed from results/ do NOT appear in "
              f"{REPORT.relative_to(REPO)}:")
        for lab, s in bad:
            print(f"    {lab}: expected to find {s!r}")
        return 1
    print(f"\nall {len(checks)} recomputed values appear in {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
