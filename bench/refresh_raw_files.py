"""Recompute a result directory's `meta.json.raw_files` from what is actually on disk.

`"raw_files": []` has shipped three times in this campaign (T002 review, reintroduced in
run_e4.py, caught again in the T003 review). Scripts now call `common.collect_raw`, but a
directory can still gain files *after* its script wrote meta.json -- ncu runs into an existing
result directory, for instance. This is the finaliser to run as the last step of a task.

    python bench/refresh_raw_files.py results/<run> [results/<run> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collect_raw, write_json  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc = 0
    for d in sys.argv[1:]:
        p = Path(d)
        mf = p / "meta.json"
        if not mf.exists():
            print(f"{d}: no meta.json")
            rc = 1
            continue
        meta = json.loads(mf.read_text())
        entries = collect_raw(p, meta.get("command", "unknown"))
        meta["raw_files"] = entries
        write_json(mf, meta)
        total = sum(e["bytes"] for e in entries)
        flagged = [e for e in entries if not e["secrets_scan"].startswith(("clean", "skipped"))]
        print(f"{d}: {len(entries)} raw files, {total/1e6:.2f} MB")
        for e in entries:
            print(f"    {e['bytes']:>12,d}  {e['secrets_scan']:<8s} {e['path']}")
        if flagged:
            print(f"  !! {len(flagged)} file(s) flagged by the secrets scan -- review before "
                  f"committing")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
