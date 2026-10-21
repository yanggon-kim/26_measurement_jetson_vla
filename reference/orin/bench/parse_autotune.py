"""T003 / E4 — which backend won each Inductor autotune block, and for which shape.

T003 asks: with Inductor's SM-count gate defeated, *"were Triton GEMM templates then chosen
for the action-expert projections?"*. That is a per-shape question, and the answer is in the
`+torch._inductor.select_algorithm` log: each block prints `AUTOTUNE <op>(<shapes>)` followed
by candidates ranked by time, the winner marked `100.0%`.

    python bench/parse_autotune.py scratch/T003/e4_cold_forced.log --out results/<run>/autotune_winners.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"^AUTOTUNE (\w+)\(([^)]*)\)\s*$")
CAND = re.compile(r"^\s+(\S+)\s+([\d.]+) ms\s+([\d.]+)%")


def parse(path: Path) -> list[dict]:
    rows, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        m = BLOCK.match(line)
        if m:
            cur = {"op": m.group(1), "shape": m.group(2).strip(), "cands": []}
            rows.append(cur)
            continue
        if cur is not None:
            c = CAND.match(line)
            if c:
                cur["cands"].append((c.group(1), float(c.group(2)), float(c.group(3))))
            elif line.strip() and not line.startswith((" ", "strides", "dtypes")):
                cur = None
    out = []
    for b in rows:
        if not b["cands"]:
            continue
        win = b["cands"][0]
        best_triton = next((c for c in b["cands"] if c[0].startswith("triton")), None)
        aten = next((c for c in b["cands"] if not c[0].startswith("triton")), None)
        # the action expert runs S=51 suffix tokens; everything else is vision/prefix
        lead = b["shape"].split("x", 1)[0].strip()
        out.append({
            "op": b["op"], "shape": b["shape"],
            "region": "action_expert" if lead == "51" else "vision_or_prefix",
            "winner": win[0],
            "winner_is_triton": win[0].startswith("triton"),
            "winner_ms": win[1],
            "best_triton_ms": best_triton[1] if best_triton else "",
            "best_aten_ms": aten[1] if aten else "",
            "n_candidates": len(b["cands"]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = parse(Path(args.log))
    if not rows:
        print("no autotune blocks found", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot = len(rows)
    triton = sum(r["winner_is_triton"] for r in rows)
    print(f"  {tot} autotune blocks, Triton won {triton} ({100*triton/tot:.1f} %)")
    for region in ("action_expert", "vision_or_prefix"):
        sub = [r for r in rows if r["region"] == region]
        if not sub:
            continue
        tw = sum(r["winner_is_triton"] for r in sub)
        print(f"    {region:18s} {len(sub):4d} blocks, Triton won {tw:3d} "
              f"({100*tw/len(sub):.1f} %)")
    print("\n  action-expert shapes — winner and the margin over the best Triton candidate:")
    agg = collections.defaultdict(list)
    for r in rows:
        if r["region"] == "action_expert":
            agg[r["shape"]].append(r)
    for shape, rs in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        w = rs[0]
        margin = ((float(w["best_triton_ms"]) / w["winner_ms"] - 1) * 100
                  if w["best_triton_ms"] else float("nan"))
        print(f"    {len(rs):3d} x {shape:24s} won by {w['winner']:6s} "
              f"{w['winner_ms']:.4f} ms; best triton {w['best_triton_ms']} ms "
              f"(+{margin:.1f} %)")
    print("\n  shapes Triton DID win:")
    for shape, n in collections.Counter(
            r["shape"] for r in rows if r["winner_is_triton"]).most_common():
        print(f"    {n:3d} x {shape}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
