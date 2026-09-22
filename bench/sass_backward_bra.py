"""T007 / E5b §3 — enumerate backward branches in a SASS dump and classify them by span.

Spec: `docs/03_instruction_level_analysis.md` §11 item 5 — for each line matching

    ^\\s*/\\*([0-9a-fA-F]+)\\*/\\s+(?:@!?P[0-9TF]\\s+)?(BRA)\\b(.*?)(?:;|$)

take the first `0x…` operand as the target; if the target is below the PC, emit
`source_pc, target_pc, span_bytes` and the role class of §3:

    32-128 B     small spin: bisection, retry, warp-vote loop
    128-512 B    medium body: index walk, partition discovery, reduction shuffle ladder
    > 1 KB       the real inner loop or an outer iteration loop
    > 10 KB      outer loop with a long (unrolled) body

A looped inner block has PC range `[BRA_target, BRA_source]`. If no long-span backward `BRA`
exists, the per-element work is fully unrolled and the inner range is the whole sequence —
which is itself the finding, and is reported rather than forced.

    python bench/sass_backward_bra.py <kernel>.sass --out <run>/sass/<kernel>_backward_bra.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

BRA = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?P[0-9TF]\s+)?(?:@U?P[0-9TF]\s+)?(BRA)\b(.*?)(?:;|$)")
HEX = re.compile(r"0[xX]([0-9a-fA-F]+)")


def role(span: int) -> str:
    if span < 32:
        return "tiny (<32 B) — probably not a loop"
    if span <= 128:
        return "small spin: bisection, retry, warp-vote loop"
    if span <= 512:
        return "medium body: index walk, partition discovery, reduction shuffle ladder"
    if span <= 10240:
        return "the real inner loop or an outer iteration loop"
    return "outer loop with a long (unrolled) body"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = Path(args.sass).read_text(errors="replace")

    rows, n_bra = [], 0
    for line in text.splitlines():
        m = BRA.match(line)
        if not m:
            continue
        n_bra += 1
        pc = int(m.group(1), 16)
        tgt = HEX.search(m.group(3) or "")
        if not tgt:
            continue
        t = int(tgt.group(1), 16)
        if t >= pc:
            continue                                   # forward branch: not a loop back-edge
        rows.append({"source_pc": f"0x{pc:x}", "target_pc": f"0x{t:x}",
                     "span_bytes": pc - t, "role": role(pc - t),
                     "inner_range": f"[0x{t:x}, 0x{pc:x}]"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_pc", "target_pc", "span_bytes", "role",
                                          "inner_range"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["span_bytes"]))

    # the dump has two physical lines per instruction (§2), so this is the instruction count
    n_lines = sum(1 for ln in text.splitlines() if re.match(r"^\s*/\*[0-9a-fA-F]+\*/", ln))
    print(f"  {n_lines} instruction lines, {n_bra} BRA, {len(rows)} BACKWARD")
    if not rows:
        print("  no backward BRA: the per-element work is fully unrolled and the inner range "
              "is the whole sequence (§3). That is the finding, not a failure.")
    for r in sorted(rows, key=lambda r: -r["span_bytes"])[:8]:
        print(f"    {r['source_pc']:>8s} -> {r['target_pc']:>8s}  {r['span_bytes']:>7d} B  "
              f"{r['role']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
