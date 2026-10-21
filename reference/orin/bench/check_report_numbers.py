"""Verify a report's headline numbers against the CSVs they claim to come from.

Both adversarial review passes of T005 found the same class of defect: a number in the prose
that appears in no committed file, usually because it survived from a superseded run. This
recomputes the numbers from `results/` and checks that the report text actually contains them,
so the defect fails loudly instead of being found by a reviewer.

    python bench/check_report_numbers.py                    # T005's E1/E2 report
    python bench/check_report_numbers.py --task=T010        # E11
    python bench/check_report_numbers.py --task=E4          # E4d + E4g + E4i
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
TASK = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--task=")), "T005")
DATE = ARGS[0] if ARGS else "20260922"
E1 = REPO / "results" / f"{DATE}_E1_peak"
E2 = REPO / "results" / f"{DATE}_E2_ae_kernels"
KC = REPO / "results" / f"{DATE}_E2_kernel_counts"
RAILS = REPO / "results" / f"{DATE}_E7_rails_check"
REPORT = REPO / "reports" / "E1_E2_microbench.md"


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(p.open()))


DATED = re.compile(r"^\d{8}[a-z]?_")


def newest(pattern: str) -> Path | None:
    """The newest DATED result directory matching `pattern`.

    A part that had to be re-measured lives beside the passes it supersedes, and not every
    part is re-measured the same number of times — E11e went through four passes while E11a
    went through two. Pinning one date for the whole task therefore reads superseded data
    for whichever part ran last. Directories not starting with a date (`evidence_…`,
    `regions_…`) are diagnostics and are never picked.
    """
    c = [d for d in sorted((REPO / "results").glob(pattern))
         if DATED.match(d.name) and not (d / "SUPERSEDED.md").exists()]
    return c[-1] if c else None



def present(v: str, text: str) -> bool:
    """Accept the value at the precision the report actually writes it.

    A figure quoted as 2.938 and recomputed as 2.94 is the same number; demanding one
    spelling makes the guard fire on formatting rather than on a wrong value. Only
    trailing precision is flexed — the leading digits must match exactly.

    Hoisted out of t010(), where it was a closure over that function's `text`, so the E4
    reports can use the same comparison instead of a second, subtly different one.
    """
    if v in text:
        return True
    try:
        f = float(v)
    except ValueError:
        return False
    nd = len(v.split(".")[1]) if "." in v else 0
    return any(f"{f:.{d}f}" in text for d in range(max(0, nd - 1), nd + 3))


def verdict(checks: list[tuple[str, str]], text: str, rep: Path) -> int:
    """Print one line per check and return 1 if any recomputed value is absent."""
    bad = [(lab, v) for lab, v in checks if not present(v, text)]
    for lab, v in checks:
        print(f"  {'ok  ' if present(v, text) else 'MISS'} {lab:38s} {v}")
    if bad:
        print(f"\n{len(bad)} value(s) recomputed from results/ do NOT appear in "
              f"{rep.relative_to(REPO)}:")
        for lab, v in bad:
            print(f"    {lab}: expected {v!r}")
        return 1
    print(f"\nall {len(checks)} recomputed values appear in {rep.relative_to(REPO)}")
    return 0


def fmt(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def t010() -> int:
    """T010's report, re-derived from the E11 CSVs."""
    rep = REPO / "reports" / "E11_pi0_hybrid_inputs.md"
    text = rep.read_text()
    checks = []
    A = newest("*_E11a_pi0_hostseg")
    B = newest("*_E11b_xfer")
    C = newest("*_E11c_pi0_handover")

    seg = [r for r in rows(A / "segment_summary.csv")
           if r["timing"] == "wall_isolated" and "gap_ms=0" in r["variant"]
           and r["layer"] == "0"]
    for name in ("rmsnorm1", "rope", "attention_gpu", "residual_rmsnorm2"):
        for style in ("eager", "compiled", "cudagraph"):
            m = [r for r in seg if r["segment"] == name and r["exec_style"] == style]
            if m:
                checks.append((f"{name}/{style}", f"{float(m[0]['median_ms']):.4f}"))
        m = [r for r in seg if r["segment"] == name and r["exec_style"] == "eager"]
        if m:
            checks.append((f"{name} kernels", str(float(m[0]["n_kernels"]))))

    hl = {(r["key"], r["exec_style"]): float(r["median_ms"])
          for r in rows(A / "host_layer_summary.csv") if r["variant"] == "layer=0"}
    for (k, st), v in hl.items():
        checks.append((f"t_host_layer {k}/{st}", f"{v:.3f}"))

    for r in rows(B / "xfer_fit.csv"):
        if r["method"] in ("pinned", "pageable"):
            checks.append((f"fit_a {r['method']}:{r['direction']}",
                           f"{float(r['fit_a_ms'])*1e3:.1f}"))
            checks.append((f"GBps {r['method']}:{r['direction']}",
                           f"{float(r['GBps_from_b']):.2f}"))
    for r in rows(B / "sync_summary.csv"):
        if r["sync"].startswith("_"):
            continue
        v = float(r["net_of_control_ms"] or r["median_ms"]) * 1e3
        checks.append((f"t_sync {r['sync']}", f"{v:.1f}"))

    meta = json.loads((C / "meta.json").read_text())
    checks += [("kv_bytes_total", f"{meta['kv_bytes_total']:,}"),
               ("kv_bytes_per_layer", f"{meta['kv_bytes_per_layer']:,}"),
               ("deepcopy ms", f"{meta['deepcopy_ms_per_step']:.2f}")]

    # E11d — the sweep's spreads and the one fit that exists
    for mode in ("eager", "reduce-overhead"):
        D = newest(f"*_E11d_pi0_{mode}_prefixsweep")
        if D is None or not (D / "prefix_sweep.csv").exists():
            continue
        by_region: dict[str, list[float]] = {}
        for r in rows(D / "prefix_sweep.csv"):
            by_region.setdefault(r["region"], []).append(float(r["median_ms"]))
        # only the regions the report leads with. The ten individual ae_step[i] rows are
        # in prefix_sweep.csv and are summarised by ae_total; demanding each one in prose
        # would make the guard fire on completeness rather than on correctness.
        for reg in ("vision", "vlm_prefix", "ae_total", "e2e"):
            vs = by_region.get(reg)
            if not vs:
                continue
            checks.append((f"E11d {mode} {reg} min", f"{min(vs):.2f}"))
            checks.append((f"E11d {mode} {reg} max", f"{max(vs):.2f}"))

    # E11e — the slowdown at each load, and the background rate it was driven at
    for mode in ("eager", "reduce-overhead"):
        E = newest(f"*_E11e_pi0_{mode}_mempressure")
        if E is None or not (E / "mempressure.csv").exists():
            continue
        for r in rows(E / "mempressure.csv"):
            if r["slowdown"] not in ("", "1.0"):
                checks.append((f"E11e {mode} {r['load_label']} slowdown",
                               f"{float(r['slowdown']):.2f}"))
            if r["bg_GBps_during"]:
                checks.append((f"E11e {mode} {r['load_label']} bg GB/s",
                               f"{float(r['bg_GBps_during']):.1f}"))

    # E11f — the corrected idle fraction, never the raw one
    F = newest("*_E11f_pi0_pipelining")
    if F and (F / "gpu_idle.csv").exists():
        for r in rows(F / "gpu_idle.csv"):
            if r.get("gpu_idle_frac_corrected"):
                checks.append((f"E11f idle frac {r['mode']} N={r['N']}",
                               f"{100*float(r['gpu_idle_frac_corrected']):.1f}"))
    if F and (F / "throughput.csv").exists():
        for r in rows(F / "throughput.csv"):
            checks.append((f"E11f {r['mode']} {r['phase']} inf/s",
                           f"{float(r['throughput_inf_per_s']):.2f}"))

    return verdict(checks, text, rep)


def e4g() -> int:
    """E4g's bottleneck tables, re-derived from the C1/C2 CSVs.

    The most mechanical of the E4 checks: `c1_bounds.csv` and `c2_eager_vs_best.csv` map
    one-to-one onto the report's two tables, so every headline number in the prose has a
    row behind it or it is wrong.

    Only the HEADLINE rows are checked, not all 36. t010() documents why: demanding every
    row makes the guard fire on a report being a summary rather than on a value being
    wrong, and a guard that cries wolf gets ignored.
    """
    rep = REPO / "reports" / "E4g_c1_c2_bottlenecks.md"
    if not rep.exists():
        print(f"{rep.relative_to(REPO)} does not exist — nothing to check")
        return 0
    text = rep.read_text()
    d = REPO / "results" / "20260923_E4g_c1_summary"
    checks: list[tuple[str, str]] = []
    if (d / "c1_bounds.csv").exists():
        br = rows(d / "c1_bounds.csv")
        # the action expert in each model, eager and best: the study's central claim is
        # about what ae_step is bound by, so those rows must survive verbatim
        for r in br:
            if r["region"] != "ae_step_0":
                continue
            checks.append((f"C1 {r['model']}/{r['mode']} ae SM%", fmt(float(r["sm_pct"]), 1)))
            checks.append((f"C1 {r['model']}/{r['mode']} ae mem%",
                           fmt(float(r["mem_pct"]), 1)))
    if (d / "c2_eager_vs_best.csv").exists():
        for r in rows(d / "c2_eager_vs_best.csv"):
            if r["region"] != "ae_step_0":
                continue
            checks.append((f"C2 {r['model']} ae d_long_scoreboard",
                           fmt(float(r["d_long_scoreboard"]), 3)))
    return verdict(checks, text, rep)


def e4d() -> int:
    """E4d's phase-balance table, re-derived from the ladder rows."""
    rep = REPO / "reports" / "E4d_phase_balance.md"
    if not rep.exists():
        print(f"{rep.relative_to(REPO)} does not exist — nothing to check")
        return 0
    text = rep.read_text()
    checks: list[tuple[str, str]] = []
    for model, d, n in (("pi0", "20260922_E4c_pi0_opt_ladder", 10),
                        ("pi05", "20260922_E4c_pi05_opt_ladder", 10),
                        ("smolvla", "20260922_E4c_smolvla_opt_ladder", 10),
                        ("groot_n17", "20260923_E4d_groot_n17_ladder", 4),
                        ("groot_n16", "20260923_E4d_groot_n16_ladder", 4),
                        ("groot_n15", "20260923_E4d_groot_n15_ladder", 4)):
        for f in sorted((REPO / "results" / d / "raw" / "rows").glob("*.json")):
            r = json.loads(f.read_text())
            if r.get("rung") != "eager_fp32" or int(r.get("N", n)) != n:
                continue
            try:
                e = float(r["e2e_median_ms"])
            except (TypeError, ValueError, KeyError):
                continue
            if e == e:
                checks.append((f"E4d {model} eager e2e", fmt(e, 2)))
    return verdict(checks, text, rep)


def e4i() -> int:
    """E4i's best-configuration table, re-derived from the ladder rows.

    The generator (bench/make_best_combo_report.py) picks the fastest rung that HAS a
    phase split; this recomputes that same choice independently, so a change to either
    the generator or the rows shows up as a mismatch rather than as agreement by
    construction.
    """
    rep = REPO / "reports" / "E4i_best_combinations.md"
    if not rep.exists():
        print(f"{rep.relative_to(REPO)} does not exist — nothing to check")
        return 0
    text = rep.read_text()
    checks: list[tuple[str, str]] = []
    for model, d, n in (("pi0", "20260922_E4c_pi0_opt_ladder", 10),
                        ("pi05", "20260922_E4c_pi05_opt_ladder", 10),
                        ("smolvla", "20260922_E4c_smolvla_opt_ladder", 10),
                        ("groot_n17", "20260923_E4d_groot_n17_ladder", 4),
                        ("groot_n16", "20260923_E4d_groot_n16_ladder", 4),
                        ("groot_n15", "20260923_E4d_groot_n15_ladder", 4)):
        cand = []
        for f in sorted((REPO / "results" / d / "raw" / "rows").glob("*.json")):
            r = json.loads(f.read_text())
            if int(r.get("N", n)) != n:
                continue
            try:
                e, a = float(r["e2e_median_ms"]), float(r["ae_total_ms"])
            except (TypeError, ValueError, KeyError):
                continue
            if e == e and a == a and e > 0:
                cand.append((e, a, r["rung"]))
        if not cand:
            continue
        e, a, rung = min(cand)
        checks.append((f"E4i {model} best e2e ({rung})", fmt(e, 2)))
        checks.append((f"E4i {model} best ae share %", fmt(100 * a / e, 1)))
        hi = max((100 * aa / ee, k) for ee, aa, k in cand)
        checks.append((f"E4i {model} max ae share % ({hi[1]})", fmt(hi[0], 1)))
    return verdict(checks, text, rep)


def main() -> int:
    if TASK == "T010":
        return t010()
    if TASK == "E4g":
        return e4g()
    if TASK == "E4d":
        return e4d()
    if TASK == "E4i":
        return e4i()
    if TASK == "E4":
        # all three, with a non-zero exit if ANY of them has a stale number
        return max(e4d(), e4g(), e4i())
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
