"""Does each C1 capture actually measure what its name says? Run before reporting.

`bench/c1_capture_ok.py` answers a narrower question — is the CSV complete. This answers
the one that actually went wrong twice in an afternoon:

1. **A `best` capture that silently profiled EAGER.** SmolVLA's compiled captures hit
   `torch._dynamo hit config.recompile_limit (8)` on `sample_actions`, dynamo fell back,
   and the resulting captures had byte-identical kernels to the eager ones. The counters
   barely moved, which read as a finding ("SmolVLA's expert does not convert") when it was
   an artefact.

2. **A region whose NVTX range covered a different phase.** GR00T's `vlm_prefix` range sat
   on `backbone.forward`, which CONTAINS vision, and `-c 40` takes the first 40 kernels of
   a range — so the prefix captures were profiling vision under the prefix's name.

Both show up the same way: two captures that should differ have the same leading kernels.
So that is the test.

    eager vs best, same region   -> identical leading kernels means the compile did nothing
    vision vs vlm_prefix, same mode -> identical leading kernels means the ranges overlap

Neither check can prove a capture correct. Both catch the specific failure that has
actually happened, cheaply, before a number reaches a report.

    python bench/c1_validity_check.py
"""
from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

LEAD = 3          # how many distinct leading kernels to compare


def leading(path: Path, n: int = 400) -> list[str]:
    out: list[str] = []
    try:
        rows = list(csv.DictReader(path.open()))[:n]
    except OSError:
        return out
    for r in rows:
        k = (r.get("Kernel Name") or "")[:55]
        if k and k not in out:
            out.append(k)
        if len(out) >= LEAD:
            break
    return out


def main() -> int:
    caps: dict[tuple[str, str, str], Path] = {}
    for f in sorted(glob.glob("results/20260923_E4g_c1_*/ncu/*_details.csv")):
        p = Path(f)
        m = re.match(r"(.+)_(eager|best)_(vision|vlm_prefix|ae_step_0|ae_total)"
                     r"_details\.csv$", p.name)
        if m:
            caps[m.groups()] = p

    problems = 0
    models = sorted({k[0] for k in caps})

    print("1. eager vs best, same region — identical leading kernels means the compile "
          "did nothing")
    for model in models:
        for region in ("vision", "vlm_prefix", "ae_step_0", "ae_total"):
            e, b = caps.get((model, "eager", region)), caps.get((model, "best", region))
            if not (e and b):
                continue
            le, lb = leading(e), leading(b)
            same = le == lb and le != []
            # vision is legitimately identical for GR00T: `best` compiles the head and the
            # LLM, never the vision tower, so those kernels are expected to match.
            expected = region == "vision" and model.startswith("groot")
            flag = "OK " if not same else ("expected" if expected else "SUSPECT")
            if same and not expected:
                problems += 1
            print(f"   {flag:8s} {model:12s} {region}")

    print()
    print("2. vision vs vlm_prefix, same mode — identical leading kernels means the "
          "ranges overlap")
    for model in models:
        for mode in ("eager", "best"):
            v, p_ = caps.get((model, mode, "vision")), caps.get((model, mode, "vlm_prefix"))
            if not (v and p_):
                continue
            same = leading(v) == leading(p_) and leading(v) != []
            if same:
                problems += 1
            print(f"   {'SUSPECT' if same else 'OK':8s} {model:12s} {mode}")

    print()
    print(f"captures checked: {len(caps)}   problems: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
