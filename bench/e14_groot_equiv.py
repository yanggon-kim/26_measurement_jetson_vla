"""T016 step 3 (row 7): GR00T N1.7 action equivalence of the savings variants, with controls.

Inputs: <G>/actions_<name>/actions.npz from bench/groot_phases.py --save-actions (50 inferences, seed
torch.manual_seed(5000 + i) per inference, actions (50, 1, 40, 132) float32).
Pairs compared (each: bitwise count / 50, MAXABS, REL = ||a - b|| / ||a|| per inference, max and median):
  controls   compiled ref vs compiled ref_rep (run-to-run, new process), eager ref vs eager ref_rep
  variants   ref vs keep1 / kvonce / keep1+kvonce, compiled (vlm_compile) and eager (eager_tf32)
  floor      eager ref vs compiled ref (what compilation alone changes)
A pair is only reported when both files exist; seeds are checked equal. Writes <G>/equiv.csv, <G>/equiv_summary.json.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PAIRS = [("control", "ref", "ref_rep"), ("control", "eager_ref", "eager_ref_rep"),
         ("variant", "ref", "keep1"), ("variant", "ref", "kvonce"), ("variant", "ref", "keep1+kvonce"),
         ("variant", "eager_ref", "eager_keep1"), ("variant", "eager_ref", "eager_kvonce"),
         ("variant", "eager_ref", "eager_keep1+kvonce"), ("floor", "eager_ref", "ref")]


def main() -> int:
    G = Path(sys.argv[1])
    rows, summ = [], []
    for kind, a, b in PAIRS:
        fa, fb = G / f"actions_{a}" / "actions.npz", G / f"actions_{b}" / "actions.npz"
        if not (fa.exists() and fb.exists()):
            print(f"  skip {a} vs {b}: missing")
            continue
        A, B = np.load(fa), np.load(fb)
        assert np.array_equal(A["seeds"], B["seeds"]), f"seeds differ {a} vs {b}"
        x, y = A["actions"].astype(np.float64), B["actions"].astype(np.float64)
        per = []
        for i in range(x.shape[0]):
            d = y[i] - x[i]
            per.append({"pair": f"{a} vs {b}", "kind": kind, "inference": i, "seed": int(A["seeds"][i]),
                        "bitwise": bool(np.array_equal(A["actions"][i], B["actions"][i])),
                        "maxabs": float(np.abs(d).max()), "rel": float(np.linalg.norm(d) / np.linalg.norm(x[i]))})
        rows += per
        s = {"pair": f"{a} vs {b}", "kind": kind, "inferences": len(per),
             "bitwise": sum(r["bitwise"] for r in per),
             "MAXABS": max(r["maxabs"] for r in per), "median_maxabs": statistics.median(r["maxabs"] for r in per),
             "REL": max(r["rel"] for r in per), "median_rel": statistics.median(r["rel"] for r in per),
             "nonfinite": int(A["nonfinite"]) + int(B["nonfinite"]),
             "compiled_a": str(A["compiled"]), "compiled_b": str(B["compiled"])}
        summ.append(s)
        print(f"{kind:8s} {s['pair']:32s} bitwise {s['bitwise']}/{s['inferences']}  MAXABS {s['MAXABS']:.4g} "
              f"(median {s['median_maxabs']:.3g})  REL {s['REL']:.4g} (median {s['median_rel']:.3g})  nonfinite {s['nonfinite']}")
    with (G / "equiv.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (G / "equiv_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
