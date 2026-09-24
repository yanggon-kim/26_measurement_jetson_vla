"""Choose E5b's target kernels from C1's committed data, applying docs/03 §1 and §5.

E5b profiles the dominant action-expert kernels at instruction level. Finding out which
kernels those are does not need a new capture: C1 already measured every kernel of
`ae_step_0` for every model in both modes, with duration and SM throughput. This reads
that and applies the document's two rules instead of choosing by eye.

  §1 gate    do NOT analyse a kernel whose Compute (SM) Throughput is below 5 % *and*
             whose duration is comparable to the launch gap. It is launch-bound, and its
             instruction mix explains nothing about why the phase is slow. The launch gap
             on this device is 62 µs, measured in E5
             (results/20260922_E5_pi0_eager_N4/timeline_summary.csv).
  §5 coverage the chosen targets must cover >= 60 % of `ae_step` kernel time.

Both conditions have to hold together: a kernel that is 30 % of the phase at 2 % SM
throughput and 500 µs is NOT launch-bound (it is far longer than the gap) and stays in.

    python bench/pick_e5b_targets.py                  # print the table
    python bench/pick_e5b_targets.py --emit-queue     # emit the shell lines to run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e5b import load_csv          # noqa: E402
from crosscheck_sass import _short_name   # noqa: E402

LAUNCH_GAP_US = 62.0        # E5, pi0 eager: median inter-kernel gap
MODELS = ("pi0", "pi05", "smolvla")
MODES = ("eager", "best")


def targets_for(model: str, mode: str, region: str = "ae_step_0",
                coverage: float = 60.0) -> tuple[list[dict], float]:
    f = Path(f"results/20260923_E4g_c1_{model}_{mode}/ncu/"
             f"{model}_{mode}_{region}_details.csv")
    if not f.exists():
        return [], 0.0
    merged: dict[str, dict] = {}
    for k in load_csv(f).values():
        n = _short_name(k["name"]) or k["name"][:40]
        e = merged.setdefault(n, {"dur": 0.0, "sm": 0.0, "n": 0})
        e["dur"] += k.get("dur_us", 0.0)
        e["sm"] = max(e["sm"], k.get("sm_pct", 0.0))
        e["n"] += 1
    total = sum(v["dur"] for v in merged.values())
    if total <= 0:
        return [], 0.0
    out, cum = [], 0.0
    for n, v in sorted(merged.items(), key=lambda kv: -kv[1]["dur"]):
        share = 100 * v["dur"] / total
        launch_bound = v["sm"] < 5.0 and v["dur"] < LAUNCH_GAP_US
        row = {"model": model, "mode": mode, "region": region, "kernel": n,
               "duration_us": round(v["dur"], 1), "launches": v["n"],
               "sm_pct": round(v["sm"], 1), "share_pct": round(share, 1),
               "verdict": "launch-bound (§1): skipped" if launch_bound else "TARGET"}
        if not launch_bound:
            cum += share
            row["cumulative_pct"] = round(cum, 1)
        out.append(row)
        if cum >= coverage:
            break
    return out, cum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=float, default=60.0)
    ap.add_argument("--emit-queue", action="store_true")
    args = ap.parse_args()

    lines = []
    for model in MODELS:
        for mode in MODES:
            rows, cum = targets_for(model, mode, coverage=args.coverage)
            if not rows:
                print(f"{model}/{mode}: no C1 capture", file=sys.stderr)
                continue
            if not args.emit_queue:
                print(f"\n{model}/{mode}  (targets cover {cum:.1f} % of ae_step "
                      f"kernel time; §5 wants >= {args.coverage:.0f} %)")
                for r in rows:
                    print(f"   {r['share_pct']:5.1f}%  {r['duration_us']:8.1f} us  "
                          f"x{r['launches']:<3d} SM {r['sm_pct']:5.1f}%  "
                          f"{r['verdict']:26s} {r['kernel']}")
            if cum < args.coverage:
                print(f"   *** coverage {cum:.1f} % is BELOW §5's {args.coverage:.0f} % "
                      f"— report the shortfall rather than lowering the bar",
                      file=sys.stderr)
            for r in rows:
                if r["verdict"] != "TARGET":
                    continue
                # a regex, because ncu matches the MANGLED name and the short name is a
                # substring of it; anchoring would never match
                lines.append(
                    f'NCU_COUNT=20 bash bench/run_ncu_e5b.sh {model} {mode} '
                    f'{r["region"]} results/${{DATE}}_E5b_{model}_{mode} '
                    f'"{re.escape(r["kernel"])}"')
    if args.emit_queue:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
