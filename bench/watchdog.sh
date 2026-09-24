#!/usr/bin/env bash
# Wake the agent after N minutes WITH the audit already rendered, so the wake-up is the
# audit and not a status glance (owner, 2026-09-23): past correctness, what is running,
# what is next. Usage: bash bench/watchdog.sh [minutes]
# Usage:
#   bash bench/watchdog.sh [minutes]           one audit after N minutes, then exit
#   bash bench/watchdog.sh [minutes] --every   audit EVERY N minutes, forever
#
# The one-shot form fires once and has to be re-armed by hand, which means the cadence
# depends on the agent remembering -- and on 2026-09-24 it drifted from every ~20 minutes
# to on-demand without anyone noticing. --every removes that dependency: one invocation
# keeps auditing at a fixed cadence until it is stopped.
MIN="${1:-25}"
EVERY="${2:-}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla

audit_once() {

echo "=================== WATCHDOG AUDIT $(date +%H:%M) ==================="
echo
echo "--- 1a. PAST CORRECTNESS: Parts A/B/C, the work that is already reported"
$HOME/venvs/pi0/bin/python - <<'PYA'
import glob, json

# A rung that leaves NO row is invisible to a per-file scan: GR00T's two backbone rungs
# crashed without writing anything and this audit called the model "4 measured, 0 failed".
# So the expected set is named here and anything absent is reported as MISSING.
LEROBOT = ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
           "vis_graphs", "vlm_compile", "vlm_graphs", "whole_graphs", "vlm_compile_nvg",
           "bf16_base", "bf16_vision", "bf16_norms", "bf16_both"]
GROOT = ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
         "vlm_compile", "vlm_graphs"]
EXPECTED = {"pi0": LEROBOT, "pi05": LEROBOT, "smolvla": LEROBOT + ["vis_static_pos"],
            "groot_n17": GROOT, "groot_n16": GROOT, "groot_n15": GROOT}
for name, pat in (("pi0",       "results/20260922_E4c_pi0_opt_ladder/raw/rows/*.json"),
                  ("pi05",      "results/20260922_E4c_pi05_opt_ladder/raw/rows/*.json"),
                  ("smolvla",   "results/20260922_E4c_smolvla_opt_ladder/raw/rows/*.json"),
                  ("groot_n17", "results/20260923_E4d_groot_n17_ladder/raw/rows/*.json"),
                  ("groot_n16", "results/20260923_E4d_groot_n16_ladder/raw/rows/*.json"),
                  ("groot_n15", "results/20260923_E4d_groot_n15_ladder/raw/rows/*.json")):
    ok, na, bad = [], [], []
    for f in sorted(glob.glob(pat)):
        r = json.load(open(f))
        # NaN is NOT a measurement, and a number stored as a STRING still is one: the
        # first version of this check miscounted SmolVLA 14/0 when the truth was 11/3,
        # and later called 8 good pi0 rungs failed. Coerce, then reject NaN and <= 0.
        try:
            fv = float(r.get("e2e_median_ms"))
            measured = fv == fv and fv > 0
        except (TypeError, ValueError):
            measured = False
        (na if (r.get("status") == "not applicable" or r.get("not_applicable_reason"))
         else ok if measured else bad).append(r["rung"])
    missing = [x for x in EXPECTED.get(name, []) if x not in set(ok) | set(na) | set(bad)]
    print(f"  {name:10s} measured={len(ok):2d}  n/a={len(na)}  FAILED={len(bad)}"
          f"  MISSING={len(missing)}"
          + (f"  -> {sorted(set(bad))}" if bad else "")
          + (f"  MISSING:{missing}" if missing else ""))

# C1: 36 captures, checked for having enough rows rather than for mere existence.
c1 = sorted(glob.glob("results/20260923_E4g_c1_*/ncu/*_details.csv"))
thin = [f for f in c1 if sum(1 for _ in open(f)) < 300]
print(f"  C1         captures={len(c1)}/36"
      + (f"  THIN (<300 rows): {len(thin)}" if thin else "  all >=300 rows"))
PYA
echo
echo "--- 1b. PAST CORRECTNESS: the phases this plan is executing now"
$HOME/venvs/pi0/bin/python - <<'PYB'
"""Not "does the directory exist" but "does it hold the number it exists for".

Every check below names the artefact AND the property that makes it usable, because the
failures this campaign actually hit were all of the silent kind: a 0-byte stdout, a CSV
without the metric it was captured for, a meta.json that never got written because the
process died between writing the CSV and writing the JSON.
"""
import csv, glob, os
from pathlib import Path


def _live_outdirs():
    """Output directories named on the command line of a process running right now.

    Without this, a run that is HALFWAY THROUGH is reported as BROKEN ("no power.csv"),
    because its CSV is only written at the end. An audit that cries wolf on healthy work
    gets ignored, which defeats the point of auditing at all. Matched on the exact argv
    elements, never on a substring of the joined cmdline.
    """
    live = set()
    me = os.getpid()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = [a.decode('utf-8', 'replace')
                    for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
        except OSError:
            continue
        for a in argv[1:]:
            if a.startswith("results/") or a.startswith("scratch/"):
                live.add(a)
    return live


LIVE = _live_outdirs()


def _is_live(dirname: str) -> bool:
    return any(dirname == v or dirname.startswith(v) for v in LIVE)


def power_ok(p, need_phases=frozenset(("prefix", "ae", "full")), minrows=18):
    f = Path(p) / "power.csv"
    if not f.exists():
        return "no power.csv"
    rows = list(csv.DictReader(f.open()))
    got = {r["phase"] for r in rows}
    if not need_phases <= got:
        return f"phases {sorted(need_phases - got)} MISSING"
    if len(rows) < minrows:
        return f"only {len(rows)} rows (expect >= {minrows})"
    if not (Path(p) / "meta.json").exists():
        return "power.csv but NO meta.json"
    if not any(Path(p, "raw").glob("tegrastats_*.log")):
        return "no tegrastats log"
    return ""


E7 = [("pi0 N=4 eager",    "results/*_E7_pi0_N4_eager_*"),
      ("pi0 N=10 eager",   "results/*_E7_pi0_N10_eager_*"),
      ("pi0 N=4 best",     "results/*_E7_pi0_N4_bf16_both_*"),
      ("pi0 N=10 best",    "results/*_E7_pi0_N10_bf16_both_*"),
      ("pi05 eager",       "results/*_E7_pi05_N10_eager_*"),
      ("pi05 best",        "results/*_E7_pi05_N10_bf16_both_graphs_*"),
      ("smolvla eager",    "results/*_E7_smolvla_N10_eager_*"),
      ("smolvla best",     "results/*_E7_smolvla_N10_vlm_graphs_*"),
      ("groot_n17 eager",  "results/*_E7_groot_n17_eager_fp32"),
      ("groot_n17 best",   "results/*_E7_groot_n17_vlm_compile"),
      ("groot_n16 eager",  "results/*_E7_groot_n16_eager_fp32"),
      ("groot_n16 best",   "results/*_E7_groot_n16_all_compile_autotune"),
      ("groot_n15 eager",  "results/*_E7_groot_n15_eager_fp32"),
      ("groot_n15 best",   "results/*_E7_groot_n15_all_compile_autotune")]
done = bad = 0
lines = []
for label, pat in E7:
    ds = sorted(glob.glob(pat))
    if not ds:
        lines.append(f"       {label:18s} not started")
        continue
    why = power_ok(ds[-1])
    if why and _is_live(ds[-1]):
        lines.append(f"       {label:18s} RUNNING NOW ({why} — expected mid-run)")
    elif why:
        bad += 1
        lines.append(f"       {label:18s} BROKEN: {why}   ({ds[-1]})")
    else:
        done += 1
print(f"  E7 power    {done}/{len(E7)} usable, {bad} BROKEN, "
      f"{len(E7) - done - bad} not started")
for line in lines[:10]:
    print(line)

# Live-aware, like the E7 check above. Without this the run that STARTED five minutes
# ago is reported BROKEN, because its output file is written at the end -- which is what
# happened to 20260924_E11f_pi0_pipelining at 13:31, five minutes into a fresh matrix.
# The E7 rows got this treatment and the three rows beside them did not, which is the
# same omission as bw_hog sitting un-fixed next to tegrastats.
for tag, pat, need, target in (
        ("E11 hybrid",  "results/*_E11*_*",  "model_inputs.csv",       4),
        ("E5 timeline", "results/*_E5_*",    "timeline_summary.csv",  12),
        ("E5b instr",   "results/*_E5b_*",   "stall_attribution.csv",  6)):
    rows = [(d, (Path(d) / need).exists() and (Path(d) / need).stat().st_size > 0)
            for d in sorted(glob.glob(pat))]
    ok = sum(1 for _, o in rows if o)
    live = [os.path.basename(d) for d, o in rows if not o and _is_live(d)]
    broken = [os.path.basename(d) for d, o in rows if not o and not _is_live(d)]
    print(f"  {tag:11s} {ok} usable of {len(rows)} present (target ~{target})"
          + (f"   RUNNING: {live}" if live else "")
          + (f"   BROKEN: {broken[:3]}" if broken else ""))

# A 0-byte stdout is a run that produced nothing while leaving a directory behind --
# UNLESS that run is still going, in which case a capture whose output is still buffered
# is entirely normal. This was the sixth false alarm of the same class today; every other
# check in this audit had already been made live-aware one at a time, which is how a
# checker accumulates them. Whole file swept now, not another instance.
empty_all = [f for f in glob.glob("results/*/raw/*stdout*") if os.path.getsize(f) == 0]
empty, empty_live = [], []
for f in empty_all:
    d = os.path.dirname(os.path.dirname(f))
    (empty_live if _is_live(d) else empty).append(os.path.basename(d))
if empty:
    print(f"  *** {len(empty)} ZERO-BYTE stdout (a run that wrote nothing): {empty[:3]}")
if empty_live:
    print(f"  ({len(empty_live)} empty stdout in a RUNNING job — output still buffered)")
PYB
echo
echo "--- 2. RUNNING NOW (from /proc; pgrep -f matches itself)"
$HOME/venvs/pi0/bin/python - <<'PYC'
import os
from pathlib import Path

# Match the EXACT argv element, never a substring of the joined cmdline. Substring
# matching has cost this project three incidents -- the shell that creates a heredoc
# contains the pattern, so the matcher finds itself; once it killed its own shell.
JOBS = ("opt_ladder.py", "groot_phases.py", "paper_repro.py", "power_phases.py",
        "phase_loops.py", "host_segments.py", "nsys_target.py", "ncu_target.py",
        "build_inst_inventory.py", "libero_eval.py",
        # added 2026-09-24: the E11 scripts. Section 2 did not know about these and
        # reported "NOTHING RUNNING" with pipelining.py live. One list, two readers.
        "handover.py", "prefix_sweep.py", "mempressure.py", "pipelining.py",
        "groot_segments.py", "consolidate_e11.py")

me = os.getpid()
ppid, argvs, hits = {}, {}, []
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    pid = int(d.name)
    try:
        argv = [a.decode('utf-8', 'replace')
                for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
        stat = (d / 'stat').read_bytes()
    except OSError:
        continue
    if not argv:
        continue
    argvs[pid] = argv
    # ppid is field 4 of /proc/<pid>/stat, AFTER the comm field, which can contain
    # spaces and parentheses -- so split on the last ')' rather than on whitespace.
    try:
        ppid[pid] = int(stat.rsplit(b')', 1)[1].split()[1])
    except (IndexError, ValueError):
        ppid[pid] = 1
    exe = argv[0].rsplit('/', 1)[-1]
    is_py = exe.startswith('python')
    is_prof = exe in ('ncu', 'nsys')
    named = any(a.rsplit('/', 1)[-1] in JOBS for a in argv[1:])
    # nsys' own helpers are not jobs: the agent daemonises (reparenting to init) and
    # nsys-tee is a pipe relay. Counting them as separate work is what produced
    # "MORE THAN ONE GPU JOB LIVE" for a single healthy capture.
    helper = ('--start-agent' in argv) or exe.startswith('nsys-tee')
    if (is_py and named) or (is_prof and not helper):
        hits.append(pid)

hitset = set(hits)


def root_of(pid):
    """The outermost matched ancestor: a capture launched BY a measurement is the same
    job, not a second one competing for the GPU."""
    seen, cur, best = set(), pid, pid
    while cur in ppid and cur not in seen:
        seen.add(cur)
        cur = ppid[cur]
        if cur in hitset:
            best = cur
    return best


for pid in sorted(hits):
    print(f"  {pid}: {' '.join(argvs[pid])[:130]}")
roots = {root_of(p) for p in hits}
if not hits:
    print("  NOTHING RUNNING  <-- if the queue is not done, this is the defect")
elif len(roots) > 1:
    print(f"  *** {len(roots)} INDEPENDENT GPU JOBS LIVE (roots {sorted(roots)}) -- they "
          f"are sharing the GPU and every number from both is confounded ***")
else:
    print(f"  one job tree (root {sorted(roots)[0]}), {len(hits)} process(es) -- healthy")
PYC
echo
echo "--- 2b. PROFILER CONCURRENCY (owner, 2026-09-23: never two at once)"
$HOME/venvs/pi0/bin/python - <<'PYD'
import os
from pathlib import Path
me = os.getpid()
ppid, sessions, helpers = {}, [], []
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    pid = int(d.name)
    try:
        argv = [a.decode('utf-8', 'replace')
                for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
        stat = (d / 'stat').read_bytes()
    except OSError:
        continue
    if not argv:
        continue
    try:
        ppid[pid] = int(stat.rsplit(b')', 1)[1].split()[1])
    except (IndexError, ValueError):
        ppid[pid] = 1
    exe = argv[0].rsplit('/', 1)[-1]
    if exe not in ('ncu', 'nsys'):
        continue
    # ONE nsys session spawns several nsys processes: the launcher, a daemonised
    # --start-agent that reparents to init, and nsys-tee. Counting processes reported
    # "TWO OR MORE PROFILERS LIVE" for a single healthy capture and would have had me
    # kill it. A session is a top-level `ncu`/`nsys` invocation with a subcommand.
    if '--start-agent' in argv or exe.startswith('nsys-tee'):
        helpers.append((pid, ' '.join(argv[:3])[:70]))
    else:
        sessions.append((pid, ' '.join(argv[:3])[:70]))

# a launcher that is a child of another session is part of it, not a second one
sess_pids = {p for p, _ in sessions}


def has_matched_ancestor(pid):
    seen, cur = set(), pid
    while cur in ppid and cur not in seen:
        seen.add(cur)
        cur = ppid[cur]
        if cur in sess_pids:
            return True
    return False


roots = [(p, c) for p, c in sessions if not has_matched_ancestor(p)]
print(f"  profiler sessions: {len(roots)}  (+{len(sessions) - len(roots)} child "
      f"launcher(s), {len(helpers)} helper(s))")
for pid, c in roots:
    print(f"    {pid}: {c}")
if len(roots) > 1:
    print("  *** TWO OR MORE PROFILER SESSIONS LIVE - STOP ONE NOW "
          "(they are root-owned; plain kill returns Operation not permitted) ***")
PYD
echo
echo "--- 2c. ORPHANS (root-owned survivors of a killed queue, which load the machine)"
ps -eo pid,user,etime,args --no-headers 2>/dev/null \
  | awk '$3 ~ /bin\/python$/ && $4 ~ /compile_worker/ {n[$2]++} END {for (u in n) printf "  %d orphaned inductor compile_worker owned by %s\n", n[u], u}'
$HOME/venvs/pi0/bin/python - <<'PYF'
"""Orphans and missing artefacts — but only where a measurement is NOT running.

The first version of these two checks fired on the run that was in progress: tegrastats
IS supposed to be running during a power measurement, and a directory has no meta.json
until its run finishes. An audit that reports healthy work as broken gets ignored.
"""
import glob, os
from pathlib import Path

JOBS = ("opt_ladder.py", "groot_phases.py", "paper_repro.py", "power_phases.py",
        "phase_loops.py", "host_segments.py", "nsys_target.py", "ncu_target.py",
        "build_inst_inventory.py", "libero_eval.py",
        # added 2026-09-24: the E11 scripts. Section 2 did not know about these and
        # reported "NOTHING RUNNING" with pipelining.py live. One list, two readers.
        "handover.py", "prefix_sweep.py", "mempressure.py", "pipelining.py",
        "groot_segments.py", "consolidate_e11.py")
live_dirs, running, hogs = set(), False, []
_all_argv = []
me = os.getpid()
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    try:
        argv = [a.decode('utf-8', 'replace')
                for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
    except OSError:
        continue
    if not argv:
        continue
    _all_argv.append(argv)
    names = [a.rsplit('/', 1)[-1] for a in argv[1:]]
    if any(n in JOBS for n in names):
        running = True
        live_dirs |= {a for a in argv[1:] if a.startswith(("results/", "scratch/"))}
    if any(n == "bw_hog.py" for n in names):
        hogs.append(d.name)

# bw_hog is mempressure's OWN background load, so it is supposed to be running while
# E11e is. Warning unconditionally reported 7 healthy hogs as orphans at 12:21 -- the
# same false alarm the tegrastats check already had fixed, in a check written right
# beside it. An orphan is a hog with no mempressure alive to own it.
mempressure_live = any(
    "mempressure.py" in [a.rsplit("/", 1)[-1] for a in argv[1:]]
    for argv in _all_argv)
if hogs and not mempressure_live:
    print(f"  *** bw_hog ORPHANED (pids {hogs}) with no mempressure running — it "
          f"saturates memory bandwidth and silently slows whatever measures next")
elif hogs:
    print(f"  ({len(hogs)} bw_hog process(es) — mempressure's own background load, "
          f"expected while E11e runs)")
if os.system("pgrep -x tegrastats >/dev/null 2>&1") == 0 and not running:
    print("  *** tegrastats is running with NO measurement — it corrupts the next "
          "power log (tegrastats --stop, then pkill -9 -x tegrastats, both under sudo)")

missing = []
for d in sorted(glob.glob("results/*_E7_*") + glob.glob("results/*_E11*")
                + glob.glob("results/*_E5*")):
    if Path(d, "meta.json").exists():
        continue
    if any(d == v or d.startswith(v) for v in live_dirs):
        continue                       # in progress: meta.json is written at the end
    missing.append(os.path.basename(d))
if missing:
    print(f"  {len(missing)} finished run(s) with NO meta.json (protocol §5): "
          f"{missing[:4]}")
PYF
echo
echo "--- 3. QUEUE MARKERS"
# glob every queue log: naming two of them by hand meant the 07:22 audit printed markers
# from a queue that had been superseded two queues earlier.
ls -t scratch/T004/queue_*.log scratch/queues/*.log 2>/dev/null | head -3 | while read -r f; do
  echo "  [$f]"
  grep -hE "_START |_DONE |^####|EXIT=[^0]" "$f" 2>/dev/null | tail -6 | sed 's/^/    /'
done
echo
echo "--- 3b. NON-ZERO EXITS, all of them, across every queue log"
# Section 3 already greps EXIT=[^0], but inside a `tail -6` window per file. The 16:22
# audit on 2026-09-24 therefore printed six healthy lines and stayed silent about
# "EXIT=1 smolvla E11d prefix sweep" four lines above them -- a real lost measurement
# that I found by reading the log by hand, not from the audit. A failure must not be
# able to scroll out of the audit that exists to catch it, so this counts ALL of them
# and prints each one.
# Scoped to scratch/queues/, THIS plan's queues. The first version also scanned
# scratch/T004/queue_*.log and reported nine failures from a task that finished days ago
# and whose failures were all resolved -- a check that cries wolf is worse than none.
# A non-zero EXIT is only OUTSTANDING if no _DONE marker follows it in the same log:
# a queue that failed, was fixed and then ran to completion has already dealt with it.
# E7's log holds six EXIT=1 lines -- every one of them a defect I fixed and re-ran green,
# 14/14 points -- and screaming about those every ten minutes is the same crying-wolf
# failure as scanning the finished T004 logs. Matching on the LABEL does not work here:
# the older run() wrote a bare "EXIT=$?" with no label at all.
_fails=0
for f in scratch/queues/*.log; do
  [ -f "$f" ] || continue
  # grep -c prints 0 and exits 1 on no match, so `|| echo 0` appended a SECOND line and
  # made every comparison "[: 0\n0: integer expression expected".
  out=$(awk '
    /^EXIT=[^0]/          { pending[++n] = FNR ": " $0 }
    /_DONE/               { n = 0 }   # E7_MAXN_DONE, E11_DONE, E11FIX_DONE: no space
                                        # before _DONE, which "/ _DONE /" did not match
    END { for (i = 1; i <= n; i++) print pending[i] }' "$f")
  [ -n "$out" ] || continue
  n=$(printf "%s\n" "$out" | wc -l)
  _fails=$((_fails + n))
  echo "  QUEUE_FAILURES=$n outstanding in $f"
  printf "%s\n" "$out" | sed 's/^/      line /'
done
if [ "$_fails" -eq 0 ]; then
  echo "  QUEUE_FAILURES=0 outstanding (earlier failures all followed by a _DONE marker)"
else
  echo "  *** QUEUE_FAILURES=$_fails outstanding -- each is a point that is NOT measured;"
  echo "      check whether the queue's have() guard will re-run it or silently skip it ***"
fi

echo
echo "--- 4. ACTION-EXPERT SHARE per configuration (owner: prefer combinations that make"
echo "        the expert LARGER -- that is where a PIM case is strongest)"
$HOME/venvs/pi0/bin/python - <<'PYE'
import glob, json
best = {}
for pat, name in (("results/20260922_E4c_pi0_opt_ladder/raw/rows/*.json", "pi0"),
                  ("results/20260922_E4c_pi05_opt_ladder/raw/rows/*.json", "pi0.5"),
                  ("results/20260922_E4c_smolvla_opt_ladder/raw/rows/*.json", "SmolVLA"),
                  ("results/20260923_E4d_groot_n17_ladder/raw/rows/*.json", "GR00T N1.7"),
                  ("results/20260923_E4d_groot_n16_ladder/raw/rows/*.json", "GR00T N1.6"),
                  ("results/20260923_E4d_groot_n15_ladder/raw/rows/*.json", "GR00T N1.5")):
    rows = []
    for f in glob.glob(pat):
        r = json.load(open(f))
        try:
            e = float(r.get("e2e_median_ms")); a = float(r.get("ae_total_ms"))
        except (TypeError, ValueError):
            continue
        if e != e or a != a or e <= 0:
            continue
        if name == "pi0" and int(r.get("N", 10)) != 10:
            continue
        rows.append((100 * a / e, e, r["rung"]))
    if rows:
        best[name] = (max(rows), min(rows, key=lambda x: x[1]))
print(f"  {'model':12s}{'max AE share':>26s}   {'lowest e2e':>26s}")
for m, (hi, lo) in best.items():
    print(f"  {m:12s}{hi[0]:8.1f} % {hi[2][:14]:15s}{lo[1]:9.2f} ms {lo[2][:14]:15s}")
PYE
echo
echo "--- 5. UPCOMING ORDER (re-derived from the plan, not remembered)"
cat <<'ORD'
    Parts A/B/C are COMPLETE and reported. What follows is the original campaign, with
    everything that needs a human reboot LAST (owner, 2026-09-24).

      1. E7 energy at MAXN, six models, eager + best     DONE 2026-09-24, 14/14 points,
                                                            reports/E7_power_energy.md
      2. E11 hybrid inputs T011/T012 (pi05, smolvla, groot)  <- RUNNING. The PIM model's
                                                            direct input; needs per-model
                                                            segment code, not a --model flag
      2b. queue_e11_fixup.sh -- CHAINED on E11_DONE. Re-runs the SmolVLA compiled-mode
          points lost to the static-posids bug (fixed in load_policy, f7b91a9) and
          re-consolidates smolvla. Do NOT re-run queue_e11.sh instead: its have() guard
          keys on the eager directory and would skip exactly what is missing.
      2c. Then, with nothing executing them: fix that have() guard, and regenerate
          meta.json for 20260924_E11a_pi05_hostseg and _E11c_pi05_handover (empty
          model_commit). Both recorded in scratch/FIXME_sweep_selfmatch.txt.
      3. E5 nsys timeline, six models (the compiled half + the busy 16 %). Also tests the
         GPU-idle hypothesis registered in reports/E7_power_energy.md.
      4. E5b / T007 instruction level, six models
      5. check_report_numbers.py extended to E4d/E4g/E4i  (CPU-only; run between GPU jobs)
      6. E7 at MODE_50W, unattended -- decline any reboot prompt and record it
      7. LAST, needs a human reboot: E7 at MODE_30W and MODE_15W. Never confirm the
         prompt, never use nvpmodel --force; write "next: resume E7 at mode <id>" at the
         top of STATUS.md and hand off with the exact commands.
ORD
}

if [ "$EVERY" = "--every" ]; then
  while true; do
    sleep $((MIN * 60))
    audit_once
  done
else
  sleep $((MIN * 60))
  audit_once
fi
