"""E7v analysis: vision / language (VLM prefill) / action energy and latency per model, MAXN, fixed sampler.

Reads every results/*_E7v_*/power.csv (bench/power_phases.py and bench/groot_phases.py with
--phases prefix,vision,vlm,ae,full). Per run, module rail: median ms and above-idle energy per inference (mJ) for each
phase; shares of vision / language / action in (vision + vlm + ae); two closure checks, both measured:
  vision + vlm vs prefix (the split loops against the unsplit prefix loop)
  prefix + ae  vs full   (the stage loops against the whole frame)
Writes <out>/e7v_split.csv and <out>/e7v_split.md.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pc in sorted(glob.glob("results/*_E7v_*/power.csv")):
        d = Path(pc).parent
        m = json.load(open(d / "meta.json"))
        P = {}
        for r in csv.DictReader(open(pc)):
            if r["rail"] == "module":
                P[r["phase"]] = (float(r["median_ms"]), float(r["energy_per_inference_above_idle_mJ"]),
                                 float(r["above_idle_mW"]))
        if not all(k in P for k in ("vision", "vlm", "ae", "prefix", "full")):
            continue
        v, l, a = P["vision"], P["vlm"], P["ae"]
        e3 = v[1] + l[1] + a[1]
        t3 = v[0] + l[0] + a[0]
        rows.append({
            "run": d.name, "model": m.get("model"), "config": m.get("rung") or m.get("mode") or "",
            "vision_ms": v[0], "language_ms": l[0], "action_ms": a[0], "full_ms": P["full"][0],
            "vision_mJ": round(v[1], 1), "language_mJ": round(l[1], 1), "action_mJ": round(a[1], 1),
            "full_mJ": round(P["full"][1], 1),
            "vision_W_above_idle": round(v[2] / 1e3, 2), "language_W_above_idle": round(l[2] / 1e3, 2),
            "action_W_above_idle": round(a[2] / 1e3, 2),
            "energy_share_vision": round(v[1] / e3, 4), "energy_share_language": round(l[1] / e3, 4),
            "energy_share_action": round(a[1] / e3, 4),
            "time_share_vision": round(v[0] / t3, 4), "time_share_language": round(l[0] / t3, 4),
            "time_share_action": round(a[0] / t3, 4),
            "closure_split_vs_prefix_energy": round((v[1] + l[1]) / P["prefix"][1], 4),
            "closure_split_vs_prefix_time": round((v[0] + l[0]) / P["prefix"][0], 4),
            "closure_stages_vs_full_energy": round((P["prefix"][1] + a[1]) / P["full"][1], 4)})
    with (out / "e7v_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md = ["| model | config | vision ms / J / W | language ms / J / W | action ms / J / W | energy share V / L / A | time share V / L / A | (V+L)/prefix energy | (prefix+A)/full energy |",
          "|---|---|---|---|---|---|---|---:|---:|"]
    for r in rows:
        md.append(f"| {r['model']} | `{r['config']}` | {r['vision_ms']:.1f} / {r['vision_mJ']/1e3:.2f} / {r['vision_W_above_idle']:.1f} | "
                  f"{r['language_ms']:.1f} / {r['language_mJ']/1e3:.2f} / {r['language_W_above_idle']:.1f} | "
                  f"{r['action_ms']:.1f} / {r['action_mJ']/1e3:.2f} / {r['action_W_above_idle']:.1f} | "
                  f"{100*r['energy_share_vision']:.0f} / {100*r['energy_share_language']:.0f} / {100*r['energy_share_action']:.0f} % | "
                  f"{100*r['time_share_vision']:.0f} / {100*r['time_share_language']:.0f} / {100*r['time_share_action']:.0f} % | "
                  f"{r['closure_split_vs_prefix_energy']:.3f} | {r['closure_stages_vs_full_energy']:.3f} |")
    (out / "e7v_split.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
