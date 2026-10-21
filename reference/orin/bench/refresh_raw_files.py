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
        # `produced_by` must be the command that actually made the file. Using the result
        # directory's own `command` for everything would claim, falsely, that the ncu
        # artifacts came out of the benchmark script.
        by_map = {}
        for sub, cmd in (("ncu", "bash bench/run_ncu_validation.sh " + str(p)),
                         ("nsys", "bash bench/run_nsys.sh " + str(p))):
            root = p / sub
            if root.exists():
                for f in root.rglob("*"):
                    if f.is_file():
                        by_map[str(f.relative_to(p))] = cmd
        if (p / "ncu" / "sysmem_miss_validation.csv").exists():
            by_map["ncu/sysmem_miss_validation.csv"] = (
                f"python bench/parse_ncu_validation.py {p}")
        for f in (p / "raw").glob("ncu_*"):
            by_map[str(f.relative_to(p))] = "bash bench/run_ncu_validation.sh " + str(p)

        # protocol §3.5/§5: a compressed file carries its compression command and the
        # original's sha256 + bytes. Those are recorded by whoever compressed it, in an
        # ORIGINALS.json beside the files; pick up every one of them.
        originals = {}
        for oj in p.rglob("ORIGINALS.json"):
            for o in json.loads(oj.read_text())["files"]:
                originals[o["compressed_name"]] = o

        entries = collect_raw(p, meta.get("command", "unknown"),
                              produced_by_map=by_map, originals=originals)
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
