"""Keep PROGRESS.md current while long runs are in flight.

Two monitors at different cadences, each owning one delimited block of the file so they
cannot clobber each other, and both taking an flock before writing:

  fast (default 120 s)  what is running right now, which rung, memory, elapsed vs budget
  slow (default 1200 s) the results table, re-derived from every result CSV on disk

Why a file and not just printing: the owner asked for one place to check, and a run that
spans a night has to be legible the next morning without scrolling a transcript.

Processes are found by reading /proc/<pid>/cmdline directly and excluding this process and
its parent. `pgrep -f <pattern>` is not used anywhere here: it matches the shell whose
command line contains the pattern, which has already made two waits report a live process
that had been dead for over an hour.

    python bench/progress_monitor.py --mode fast --interval 120 &
    python bench/progress_monitor.py --mode slow --interval 1200 &
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402

PROG = REPO / "PROGRESS.md"
LOCK = REPO / "scratch" / ".progress.lock"
START = datetime(2026, 9, 22, 18, 35).timestamp()
BUDGET_H = 26.0
WATCH = ("opt_ladder.py", "paper_repro.py", "phase_utilisation.py", "libero_eval.py",
         "mempressure.py", "pipelining.py", "host_segments.py", "run_ncu_regions.sh")


def running() -> list[tuple[int, str, float, float]]:
    """(pid, short cmd, elapsed_s, rss_GiB) for the scripts we care about."""
    me, parent = os.getpid(), os.getppid()
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        if pid in (me, parent):
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "progress_monitor" in cmd:
            continue
        hit = next((w for w in WATCH if w in cmd), None)
        if not hit:
            continue
        # a shell that merely mentions the script is not the script
        if not re.search(r"(python\S*|bash)\s+\S*" + re.escape(hit), cmd):
            continue
        try:
            st = (d / "stat").read_text().rsplit(")", 1)[1].split()
            start_ticks = int(st[19])
            elapsed = time.time() - (
                Path("/proc/stat").read_text().split("btime ")[1].split()[0]
                and float(Path("/proc/stat").read_text().split("btime ")[1].split()[0])
                + start_ticks / os.sysconf("SC_CLK_TCK"))
            rss = int((d / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except Exception:                                          # noqa: BLE001
            elapsed, rss = 0.0, 0
        arg = re.search(r"(?:python\S*|bash)\s+(\S*" + re.escape(hit) + r"[^\n]{0,70})", cmd)
        out.append((pid, arg.group(1) if arg else hit, elapsed, rss / 2**30))
    return sorted(out, key=lambda r: r[0])


def mem() -> dict:
    mi = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = ln.partition(":")
        mi[k] = int(v.split()[0]) if v.split() else 0
    return {"avail": mi.get("MemAvailable", 0) / 1048576,
            "swap": (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / 1048576}


def tail_logs(n: int = 3) -> list[str]:
    lines = []
    logs = sorted(REPO.glob("scratch/T004/*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)[:2]
    for lg in logs:
        try:
            txt = [x for x in lg.read_text(errors="replace").splitlines()
                   if x.strip() and not x.startswith(("WARNING", "Loading", "✓"))]
        except OSError:
            continue
        for x in txt[-n:]:
            lines.append(f"`{lg.name}` · {x.strip()[:110]}")
    return lines


def fast_block() -> str:
    el = (time.time() - START) / 3600
    m = mem()
    procs = running()
    L = [f"**Updated {datetime.now():%H:%M:%S}** · elapsed **{el:.2f} h** of {BUDGET_H:.0f} "
         f"({100*el/BUDGET_H:.0f} %) · {m['avail']:.1f} GiB free · {m['swap']:.2f} GiB swap",
         ""]
    if procs:
        L.append("| pid | running | for | rss |")
        L.append("|---|---|---:|---:|")
        for pid, cmd, e, rss in procs:
            L.append(f"| {pid} | `{cmd}` | {e/60:.0f} min | {rss:.1f} GiB |")
    else:
        L.append("> **nothing is running.** If this persists, the chain broke — check the "
                 "last log below and start the next item.")
    tl = tail_logs()
    if tl:
        L += ["", "Latest log lines:", ""] + [f"- {x}" for x in tl]
    return "\n".join(L)


def slow_block() -> str:
    rows = []
    for csvf in sorted(REPO.glob("results/*/opt_ladder.csv")):
        for r in csv.DictReader(csvf.open()):
            if not r.get("e2e_median_ms"):
                continue
            try:
                e = float(r["e2e_median_ms"])
                ae = float(r["ae_total_ms"]) if r.get("ae_total_ms") else None
            except ValueError:
                continue
            rows.append({"model": csvf.parent.name.split("_")[2] if "_" in csvf.parent.name
                         else "?", "N": r["N"], "rung": r["rung"], "e2e": e,
                         "vision": r.get("vision_ms", ""), "vlm": r.get("vlm_prefix_ms", ""),
                         "ae": ae, "share": (100 * ae / e) if ae else None})
    for pc in sorted(REPO.glob("results/*_paper_config_*/paper_comparison.csv")):
        d = {x["region"]: x for x in csv.DictReader(pc.open())}
        if "e2e" not in d:
            continue
        e = float(d["e2e"]["median_ms"])
        ae = float(d["ae_total"]["median_ms"]) if "ae_total" in d else None
        rows.append({"model": pc.parent.name.split("_")[2], "N": "10",
                     "rung": "paper config (eager+TF32)", "e2e": e,
                     "vision": d.get("vision", {}).get("median_ms", ""),
                     "vlm": d.get("vlm_prefix", {}).get("median_ms", ""),
                     "ae": ae, "share": (100 * ae / e) if ae else None})
    if not rows:
        return f"*(no results yet — {datetime.now():%H:%M})*"
    L = [f"**Re-derived from every result CSV on disk at {datetime.now():%H:%M}** "
         f"({len(rows)} configurations)", "",
         "| model | N | configuration | vision | vlm_prefix | ae_total | e2e | **AE share** |",
         "|---|---:|---|---:|---:|---:|---:|---:|"]
    def f(x):
        try:
            return f"{float(x):.2f}"
        except (TypeError, ValueError):
            return "—"
    for r in rows:
        L.append(f"| {r['model']} | {r['N']} | {r['rung']} | {f(r['vision'])} | "
                 f"{f(r['vlm'])} | {f(r['ae'])} | {r['e2e']:.2f} | "
                 f"{(f'{r[chr(115)+chr(104)+chr(97)+chr(114)+chr(101)]:.1f} %') if r['share'] else '—'} |")
    return "\n".join(L)


def write_block(mode: str, body: str) -> None:
    beg, end = f"<!-- {mode.upper()}:BEGIN -->", f"<!-- {mode.upper()}:END -->"
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            s = PROG.read_text()
            if beg not in s or end not in s:
                return
            i, j = s.index(beg) + len(beg), s.index(end)
            PROG.write_text(s[:i] + "\n" + body + "\n" + s[j:])
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def publish() -> str:
    """Commit and push PROGRESS.md so it is visible off this machine.

    The monitors update the working file, but AGENTS.md makes the repository the only
    channel: an owner looking at the remote sees whatever was last committed. Without this
    the file updated every two minutes locally and looked frozen to everyone else.

    Only PROGRESS.md is staged, so this cannot sweep up work in progress. Failures are
    reported and never raise: a monitor must not be able to break a measurement run.
    """
    import subprocess
    def sh(*a):
        return subprocess.run(a, cwd=REPO, capture_output=True, text=True, timeout=600)
    try:
        if not sh("git", "diff", "--quiet", "--", "PROGRESS.md").returncode:
            return "no change"
        sh("git", "add", "PROGRESS.md")
        c = sh("git", "commit", "-m",
               f"PROGRESS: {datetime.now():%H:%M} tick (bench/progress_monitor.py --mode slow)")
        if c.returncode:
            return f"commit failed: {c.stderr.strip()[:80]}"
        sh("git", "pull", "--rebase", "-q")
        r = sh("git", "push")
        return "pushed" if r.returncode == 0 else f"push failed: {r.stderr.strip()[:80]}"
    except Exception as e:                                          # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fast", "slow"), required=True)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    iv = args.interval or (120 if args.mode == "fast" else 1200)
    fn = fast_block if args.mode == "fast" else slow_block
    while True:
        try:
            write_block(args.mode, fn())
        except Exception as e:                                      # noqa: BLE001
            write_block(args.mode, f"*(monitor error: {type(e).__name__}: {str(e)[:150]})*")
        if args.mode == "slow":
            st = publish()
            write_block("slow", (fn() + f"\n\n*(published: {st} at "
                                 f"{datetime.now():%H:%M})*"))
        if args.once:
            return 0
        time.sleep(iv)


if __name__ == "__main__":
    raise SystemExit(main())
