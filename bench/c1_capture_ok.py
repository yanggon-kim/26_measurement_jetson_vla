"""Is this C1 capture actually usable? Exit 0 if yes, 1 if it must be re-run.

A file existing is not the same as a capture being complete. pi0's eager vlm_prefix showed
why: an ORPHANED ncu from a killed queue, running the previous version of the script,
finished after its directory had been moved aside and wrote a .ncu-rep.zst plus a CSV with
**0 stall metrics and 130 rows** into the fresh directory. The queue's `[ -f ... ]` check
then skipped that capture as "already done", and it would have reached the analysis as a
phase with no stall breakdown at all.

So "done" means: the CSV exists, carries the 14 stall metrics C1 asks for, and has enough
rows to be more than a header plus a rule or two.

    python bench/c1_capture_ok.py <details.csv>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

MIN_STALL_METRICS = 14      # all of smsp__average_warps_issue_stalled_*
MIN_ROWS = 300              # a real capture of 40 kernels x 4 sections is ~1300-1900


def check(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "no CSV"
    try:
        rows = list(csv.DictReader(path.open()))
    except Exception as e:                                          # noqa: BLE001
        return False, f"unreadable: {type(e).__name__}"
    if len(rows) < MIN_ROWS:
        return False, f"only {len(rows)} rows (< {MIN_ROWS})"
    stalls = {r["Metric Name"] for r in rows
              if r.get("Metric Name", "").startswith(
                  "smsp__average_warps_issue_stalled_")}
    if len(stalls) < MIN_STALL_METRICS:
        return False, f"only {len(stalls)} stall metrics (< {MIN_STALL_METRICS})"
    return True, f"{len(rows)} rows, {len(stalls)} stall metrics"


def main() -> int:
    p = Path(sys.argv[1])
    ok, why = check(p)
    print(f"  {'OK' if ok else 'REDO'}: {p.name} — {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
