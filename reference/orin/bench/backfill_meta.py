"""Back-fill meta.json for result directories written before the §5 schema was enforced.

Protocol §5 requires a meta.json per run directory, and AGENTS.md requires every committed
file to be listed there with path, sha256 and bytes. The 2026-09-25 watchdog audit found 33
directories without one -- a number that had itself been under-reported as 7, because
watchdog.sh §2c globbed only *_E7_*, *_E11* and *_E5*.

WHAT THIS DOES AND DOES NOT DO, because a back-filled meta.json is exactly the kind of file
that could quietly launder a guess into the record:

  RECOVERED, from evidence in the directory or in git:
    * the exact command, when the directory holds the capture's own .sh wrapper (ncu writes
      one per capture; it carries the environment and the full argv) or a summary.json with
      a `command` field;
    * the full file inventory with sha256 and bytes, computed now from the files themselves;
    * the commit that ADDED the directory to git, via `git log --diff-filter=A`, which is a
      fact about this repository and not an inference about the run;
    * the run's date, from summary.json's own field where it has one, else file mtimes,
      LABELLED as mtime-derived.

  NOT RECOVERED, and therefore explicitly null with a reason attached:
    * power mode, GPU/CPU clocks, temperature and throttling state. These were never
      written by the run, and nothing in the directory records them. The campaign ran at
      MAXN throughout, but "the campaign ran at MAXN" is not a per-run measurement and is
      not written into a field that is supposed to hold one.
    * the software versions in force at the time. env/ records them for the campaign; this
      script does not copy them into a per-run field as though the run had recorded them.

A field this script cannot source is `null` with its reason in `_unrecoverable`. It never
guesses, and it never overwrites an existing meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

UNRECOVERABLE = {
    "device_state": (
        "Power mode, GPU/CPU clock frequencies, temperature and throttling state were not "
        "recorded by the run that wrote this directory and are not derivable from its "
        "contents. Per AGENTS.md a latency without its power mode is not a result: this "
        "field is left null rather than filled from the campaign-wide assumption that the "
        "device was at MAXN. Runs from 2026-09-24 onward carry a protocol §1 gate block."),
    "software_versions": (
        "Not recorded per-run. env/ documents the campaign's environment, but copying it "
        "here would assert that this run observed those versions, which was never checked."),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_added_commit(d: Path) -> dict | None:
    """The commit that first added files under `d`. A fact about the repo, not the run."""
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H|%cI|%s", "--", str(d)],
            cwd=REPO, capture_output=True, text=True, timeout=60).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    h, when, subj = out.splitlines()[-1].split("|", 2)
    return {"commit": h, "committed_at": when, "subject": subj,
            "_how": "git log --diff-filter=A on this directory: the commit that first "
                    "added its files. Bounds the run from above; it is not the run's own "
                    "recorded code commit."}


def recover_commands(d: Path) -> dict:
    """Exact commands, where the directory kept them. ncu writes a .sh per capture."""
    cmds, srcs = [], []
    for sh in sorted(d.rglob("*.sh")):
        txt = sh.read_text(errors="replace")
        for line in txt.splitlines():
            if line.startswith("exec "):
                cmds.append({"capture": sh.relative_to(d).as_posix(),
                             "command": line[5:].strip()})
                break
        else:
            cmds.append({"capture": sh.relative_to(d).as_posix(),
                         "command": txt.strip()})
        srcs.append(sh.relative_to(d).as_posix())
    sj = d / "summary.json"
    if sj.exists():
        try:
            j = json.loads(sj.read_text())
        except json.JSONDecodeError:
            j = {}
        if j.get("command"):
            cmds.append({"capture": "summary.json", "command": j["command"]})
            srcs.append("summary.json")
    return {"commands": cmds, "_source": srcs}


def run_date(d: Path) -> dict:
    sj = d / "summary.json"
    if sj.exists():
        try:
            j = json.loads(sj.read_text())
        except json.JSONDecodeError:
            j = {}
        if j.get("date_utc"):
            return {"date_utc": j["date_utc"], "_how": "summary.json's own date_utc field"}
    files = [p for p in d.rglob("*") if p.is_file()]
    if not files:
        return {"date_utc": None, "_how": "directory is empty: no mtime to derive from"}
    newest = max(p.stat().st_mtime for p in files)
    return {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)),
            "_how": "MTIME-DERIVED from the newest file, not a recorded run timestamp. "
                    "A later rewrite of any file would move it."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the files; without it, report what would be written")
    ap.add_argument("--glob", default="results/*")
    args = ap.parse_args()

    wrote = skipped = empty = 0
    for d in sorted((REPO / args.glob.split("/")[0]).glob(args.glob.split("/", 1)[1])):
        if not d.is_dir():
            continue
        if (d / "meta.json").exists():
            skipped += 1
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file())
        cmds = recover_commands(d)
        meta = {
            "_schema": "protocol §5, BACK-FILLED -- read _backfill before using this file",
            "_backfill": {
                "backfilled": True,
                "backfilled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "by": "bench/backfill_meta.py",
                "why": ("this directory was written before the §5 meta.json requirement was "
                        "enforced on every experiment. The 2026-09-25 audit found 33 such "
                        "directories."),
                "reliability": ("Fields under `recovered` are sourced from evidence in this "
                                "directory or from git, each with a `_how`/`_source`. Fields "
                                "under `_unrecoverable` were never recorded and are null. "
                                "NOTHING here was inferred from expectation."),
            },
            "run": d.name,
            "recovered": {
                "date": run_date(d),
                "command": cmds,
                "git_added": git_added_commit(d),
                "files": [{"path": p.relative_to(d).as_posix(),
                           "bytes": p.stat().st_size,
                           "sha256": sha256(p)} for p in files],
                "file_count": len(files),
                "total_bytes": sum(p.stat().st_size for p in files),
            },
            "_unrecoverable": UNRECOVERABLE,
        }
        if not files:
            empty += 1
            meta["_backfill"]["note_empty"] = (
                "This directory contains NO FILES. It is not a run: it is a directory "
                "created by a queue whose run did not execute or produced nothing. It is "
                "not in git (git cannot store an empty directory), so it has never been "
                "published. Check the owning queue script and the commit log before "
                "treating its name as a result.")
        if args.apply:
            (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        n = len(meta["recovered"]["command"]["commands"])
        print(f"  {'wrote' if args.apply else 'would write'} {d.name:38s} "
              f"files={len(files):3d} commands_recovered={n}"
              f"{'  <-- EMPTY DIR' if not files else ''}")
        wrote += 1
    print(f"\n  {wrote} back-filled, {skipped} already had meta.json, {empty} empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
