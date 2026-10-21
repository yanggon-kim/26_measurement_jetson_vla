"""E7 across power modes (T004 step 4b): MAXN / MODE_50W / MODE_30W / MODE_15W side by side.

Every committed `results/*_E7_*` directory with a power.csv is grouped by (model, rung, N, matmul precision) --
the configuration, never the directory name -- and each group gets one row per phase with, per power mode:
median latency (ms) and ABOVE-IDLE module energy per inference (mJ).

Energy source, because the sampler changed (T016 step 4 / EV-10): runs made BEFORE the fix (MAXN, 50W, 30W) started
the rail sampler before warm-up, which understated compiled phases by 5-65 %. For those the energy is the
timed-window re-analysis of their own committed raw samples
(`results/20260926_E7r_pi0_energy_reconcile_MAXN/E7_all_dirs_timed_window.csv`, column E_timed_window_mJ). Runs made
with the fixed sampler (bench commit cf62176 onward: every MODE_15W run) use their power.csv directly. The source is
printed per cell (w = timed window, d = direct). Latencies are unaffected by the sampler.

The four modes are different perturbations (E0): 50W = clocks only (16 SMs, EMC free); 30W = 8 SMs, GPU 612 MHz,
8 CPUs @ 1.73 GHz; 15W = 6 SMs, GPU 408 MHz, 4 CPUs @ 1.11 GHz AND EMC capped at 2133 MHz. Nothing is averaged
across modes.

    python bench/e7_capped_compare.py --out results/<date>_E7_capped_compare
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

TW = "results/20260926_E7r_pi0_energy_reconcile_MAXN/E7_all_dirs_timed_window.csv"
MODES = ("MAXN", "MODE_50W", "MODE_30W", "MODE_15W")
FIX_DATE = "20260927"          # every run dated 2026-09-27 or later used the fixed sampler


def mode_of(pm: str) -> str:
    for m in MODES[1:]:
        if m in pm:
            return m
    return "MAXN" if "MAXN" in pm else pm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tw = {(r["dir"], r["phase"]): float(r["E_timed_window_mJ"]) for r in csv.DictReader(open(TW))}
    groups: dict = {}
    for d in sorted(glob.glob("results/2026092*_E7_*")):
        pc, mj = Path(d) / "power.csv", Path(d) / "meta.json"
        if not (pc.exists() and mj.exists()) or "rails_check" in d:
            continue
        m = json.load(open(mj))
        model = m.get("model") or ""
        rung = m.get("rung") or m.get("mode") or "eager"
        if rung in ("none (eager)",):
            rung = "eager"
        n = m.get("num_denoise_steps") or m.get("num_inference_timesteps")
        prec = m.get("matmul_precision") or ("as the rung sets it" if rung != "eager" else "default")
        pm = mode_of(str(m.get("power_mode") or (m.get("gate") or {}).get("power_mode") or ""))
        base = os.path.basename(d)
        fixed = base[:8] >= FIX_DATE
        for r in csv.DictReader(open(pc)):
            if r["rail"] != "module":
                continue
            ph = r["phase"]
            e_d = float(r["energy_per_inference_above_idle_mJ"])
            if fixed:
                e, src = e_d, "d"
            elif (base, ph) in tw:
                e, src = tw[(base, ph)], "w"
            else:
                e, src = None, "-"
            key = (model, rung, n, prec)
            groups.setdefault(key, {}).setdefault(ph, {})[pm] = {
                "ms": float(r["median_ms"]), "mJ": e, "src": src, "dir": base,
                "mW_total": float(r["mean_mW"]), "mW_above": float(r["above_idle_mW"])}
    rows = []
    for (model, rung, n, prec), P in sorted(groups.items(), key=lambda x: [str(v) for v in x[0]]):
        for ph in ("prefix", "ae", "full"):
            if ph not in P:
                continue
            row = {"model": model, "rung": rung, "N": n, "matmul": prec, "phase": ph}
            for md in MODES:
                c = P[ph].get(md)
                row[f"{md}_ms"] = round(c["ms"], 2) if c else ""
                row[f"{md}_mJ_above_idle"] = round(c["mJ"], 1) if c and c["mJ"] is not None else ""
                row[f"{md}_src"] = c["src"] if c else ""
                row[f"{md}_dir"] = c["dir"] if c else ""
            ref = P[ph].get("MAXN")
            for md in MODES[1:]:
                c = P[ph].get(md)
                ok = ref and c and ref["mJ"] and c["mJ"]
                row[f"{md}_latency_x"] = round(c["ms"] / ref["ms"], 3) if (ref and c) else ""
                row[f"{md}_energy_x"] = round(c["mJ"] / ref["mJ"], 3) if ok else ""
                row[f"{md}_EDP_x"] = round((c["mJ"] * c["ms"]) / (ref["mJ"] * ref["ms"]), 3) if ok else ""
            c30, c15 = P[ph].get("MODE_30W"), P[ph].get("MODE_15W")
            ok = c30 and c15 and c30["mJ"] and c15["mJ"]
            row["15W_vs_30W_latency_x"] = round(c15["ms"] / c30["ms"], 3) if (c30 and c15) else ""
            row["15W_vs_30W_energy_x"] = round(c15["mJ"] / c30["mJ"], 3) if ok else ""
            rows.append(row)
    with (out / "e7_power_modes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md = ["| model | rung | N | phase | MAXN ms / mJ | 50W ms / mJ | 30W ms / mJ | 15W ms / mJ | 15W vs 30W latency × / energy × |",
          "|---|---|---:|---|---|---|---|---|---|"]
    for r in rows:
        cell = lambda m: (f"{r[m + '_ms']} / {r[m + '_mJ_above_idle']}{r[m + '_src']}" if r[m + "_ms"] != "" else "—")  # noqa: E731
        md.append(f"| {r['model']} | `{r['rung']}` ({r['matmul']}) | {r['N']} | {r['phase']} | {cell('MAXN')} | "
                  f"{cell('MODE_50W')} | {cell('MODE_30W')} | {cell('MODE_15W')} | "
                  f"{r['15W_vs_30W_latency_x'] or '—'} / {r['15W_vs_30W_energy_x'] or '—'} |")
    (out / "e7_power_modes.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
