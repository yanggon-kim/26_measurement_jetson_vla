"""Regenerate floor.csv in existing E3 result directories from their own meta.json.

Used once when the ae_step compute floor was corrected to include attention FLOPs. It reads
each run's recorded `region_parameter_bytes` and `tokens_detail`, so the regenerated floors
still derive from the model as it was loaded for that run — nothing is re-measured.

    python bench/regen_floor.py results/20260921_E3_pi0_eager_N*
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e3 import floor_rows  # noqa: E402


def main(argv: list[str]) -> int:
    for d in argv:
        run = Path(d)
        m = json.loads((run / "meta.json").read_text())
        rows = floor_rows(m["region_parameter_bytes"], m["tokens_detail"])
        with (run / "floor.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"regenerated {run/'floor.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
