"""T016 step 2 (rows 5-6): seeded LIBERO N sweep analysis over results/*_E8s_pi0_<suite>_N<n>_seeded*.

Per suite and pooled, per N arm: successes, Wilson 95 % interval, exact paired McNemar vs N = 10 (two-sided,
binomial on the discordant pairs), paired difference (N - N10) with a 95 % Wald interval for paired proportions,
forward time (inference_ms_median, a select_action call > 10 ms). Episodes are paired on (task, episode) and
checked to share the same noise_seed. Repeat control: episode agreement of N10 vs N10_rep.
Writes <out>/e8s_table.csv and <out>/e8s_summary.json; prints the table.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_diff(A, B):
    """A, B: aligned 0/1 lists (arm, N10). Returns diff, lo, hi, b (arm only), c (N10 only)."""
    n = len(A)
    b = sum(1 for x, y in zip(A, B) if x and not y)
    c = sum(1 for x, y in zip(A, B) if y and not x)
    d = (b - c) / n
    var = ((b + c) / n - d * d) / n
    h = 1.959964 * math.sqrt(max(var, 0.0))
    return d, d - h, d + h, b, c


def load(d: Path):
    rows = list(csv.DictReader(open(d / "episodes.csv")))
    return {(r["task"], r["episode"]): (int(r["success"]), r["noise_seed"]) for r in rows}, \
        json.loads((d / "summary.json").read_text())


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results"
    arms = {}
    for d in sorted(REPO.glob("results/*_E8s_pi0_*_N*_seeded*")):
        if not (d / "summary.json").exists():
            continue
        m = re.search(r"_E8s_pi0_(libero_\w+?)_N(\d+)_seeded(_\w+)?$", d.name)
        if not m:
            continue
        suite, n, tag = m.group(1), int(m.group(2)), (m.group(3) or "").lstrip("_")
        arms[(suite, n, tag)] = (d, *load(d))
    rows, pooled = [], {}
    for suite in SUITES:
        ref = arms.get((suite, 10, ""))
        for (s, n, tag), (d, ep, sm) in sorted(arms.items(), key=lambda x: (x[0][1], x[0][2])):
            if s != suite:
                continue
            keys = sorted(ep)
            k = sum(ep[x][0] for x in keys)
            lo, hi = wilson(k, len(keys))
            r = {"suite": suite, "N": n, "tag": tag, "dir": d.name, "episodes": len(keys), "successes": k,
                 "rate": k / len(keys), "wilson_lo": lo, "wilson_hi": hi,
                 "forward_ms_median": sm.get("inference_ms_median"), "forward_calls": sm.get("inference_calls"),
                 "nonfinite": sm.get("nonfinite_action_elements"), "prefix_variant": sm.get("prefix_variant", "pad")}
            if ref and not (n == 10 and tag == ""):
                rep = ref[1]
                assert set(rep) == set(ep), f"{d.name}: episode keys differ from N10"
                bad = [x for x in keys if rep[x][1] != ep[x][1]]
                A = [ep[x][0] for x in keys]; B = [rep[x][0] for x in keys]
                dd, dlo, dhi, b, c = paired_diff(A, B)
                r.update({"seeds_match_N10": not bad, "only_arm": b, "only_N10": c,
                          "mcnemar_p": mcnemar_exact(b, c), "diff": dd, "diff_lo": dlo, "diff_hi": dhi,
                          "agreement": sum(x == y for x, y in zip(A, B)) / len(keys)})
                if tag == "" and r.get("prefix_variant", "pad") == "pad":
                    P = pooled.setdefault(n, {"A": [], "B": [], "suites": []})
                    P["A"] += A; P["B"] += B; P["suites"].append(suite)
            rows.append(r)
    pool = []
    for n, P in sorted(pooled.items()):
        dd, dlo, dhi, b, c = paired_diff(P["A"], P["B"])
        k, kr, m = sum(P["A"]), sum(P["B"]), len(P["A"])
        pool.append({"N": n, "suites": P["suites"], "episodes": m, "successes": k, "N10_successes": kr,
                     "wilson": wilson(k, m), "only_arm": b, "only_N10": c, "mcnemar_p": mcnemar_exact(b, c),
                     "diff": dd, "diff_lo": dlo, "diff_hi": dhi})
    with (out / "e8s_table.csv").open("w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    (out / "e8s_summary.json").write_text(json.dumps({"arms": rows, "pooled_vs_N10": pool}, indent=1) + "\n")
    md = ["| suite | N | successes / 100 [Wilson 95 %] | vs N = 10: only arm / only N10 | exact McNemar p | "
          "paired diff (pp) [95 %] | agreement | forward ms (median) |", "|---|---:|---|---:|---:|---|---:|---:|"]
    for r in rows:
        lab = f"{r['N']}" + (f" {r['tag']}" if r["tag"] else "")
        cmp = (f"{r['only_arm']} / {r['only_N10']} | {r['mcnemar_p']:.3g} | {r['diff']*100:+.1f} "
               f"[{r['diff_lo']*100:+.1f}, {r['diff_hi']*100:+.1f}] | {r['agreement']:.2f}"
               if "diff" in r else "— | — | — | —")
        md.append(f"| {r['suite']} | {lab} | **{r['successes']}** [{r['wilson_lo']*100:.1f}, {r['wilson_hi']*100:.1f}] "
                  f"| {cmp} | {r['forward_ms_median']} |")
    md += ["", "| N | suites | successes [Wilson 95 %] | N = 10 | only arm / only N10 | exact McNemar p | "
           "paired diff (pp) [95 %] |", "|---:|---:|---|---:|---:|---:|---|"]
    for p in pool:
        md.append(f"| {p['N']} | {len(p['suites'])} | **{p['successes']}/{p['episodes']}** "
                  f"[{p['wilson'][0]*100:.1f}, {p['wilson'][1]*100:.1f}] | {p['N10_successes']} | "
                  f"{p['only_arm']} / {p['only_N10']} | {p['mcnemar_p']:.3g} | {p['diff']*100:+.1f} "
                  f"[{p['diff_lo']*100:+.1f}, {p['diff_hi']*100:+.1f}] |")
    (out / "e8s_tables.md").write_text("\n".join(md) + "\n")
    for r in rows:
        extra =(f"  vs N10: +{r['only_arm']}/-{r['only_N10']} p={r['mcnemar_p']:.3g} diff {r['diff']*100:+.1f} pp "
                 f"[{r['diff_lo']*100:+.1f}, {r['diff_hi']*100:+.1f}] agree {r['agreement']:.2f} seeds_ok={r['seeds_match_N10']}"
                 if "diff" in r else "")
        print(f"{r['suite']:15s} N={r['N']:2d}{('_'+r['tag']) if r['tag'] else '':5s} {r['successes']:3d}/{r['episodes']} "
              f"[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}] fwd {r['forward_ms_median']} ms{extra}")
    for p in pool:
        print(f"POOLED N={p['N']:2d} ({len(p['suites'])} suites) {p['successes']}/{p['episodes']} vs N10 {p['N10_successes']}  "
              f"+{p['only_arm']}/-{p['only_N10']} p={p['mcnemar_p']:.3g} diff {p['diff']*100:+.1f} pp "
              f"[{p['diff_lo']*100:+.1f}, {p['diff_hi']*100:+.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
