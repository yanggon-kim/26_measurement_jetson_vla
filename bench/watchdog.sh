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
# Protocol §1: a latency without its power mode is not a result. The device is not always in
# MAXN any more -- MODE_50W runs unattended -- and a run started in the wrong mode produces
# numbers that look ordinary and are mislabelled. So every audit states the mode FIRST, and
# says loudly when it is not MAXN, because that is the condition under which a mistake is
# silent. MODE_50W keeps all 16 SMs and only lowers the clocks (816.0 vs 1300.5 MHz), so
# nothing crashes to warn you.
_pm=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
_sm=$(ls -d /sys/class/devfreq/17000000.gpu 2>/dev/null >/dev/null && echo ok || echo "")
case "$_pm" in
  *MAXN*) echo "  POWER MODE: $_pm  (the default; MAXN-class runs are valid)" ;;
  "")     echo "  *** POWER MODE UNREADABLE -- do not start a timed run until it is known ***" ;;
  *)      echo "  *** POWER MODE IS CAPPED: $_pm -- any run started now carries THIS mode,"
          echo "      not MAXN. Nothing will crash: 50W keeps 16 SMs and only lowers clocks."
          echo "      Check the run directory name and meta.json record the mode. ***" ;;
esac
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

# Failures that are ALREADY measured, explained and reported. They are still shown, as
# KNOWN rather than FAILED, so the FAILED= alarm keeps meaning "something new".
#
# Without this the audit raised groot_n17 vis_compile every ten minutes forever, for a
# defect written up months-deep in STATUS.md:214 and E4_PARTS_ABC_GUIDE.md:284. That is
# the crying-wolf failure this file has now hit three times (the stale JOBS list, the
# finished T004 queue logs, and this), and an alarm that always fires is one nobody reads.
#
# The bar for adding an entry is deliberately high: the failure must be explained IN a
# committed report, and the reference goes in the value so it can be checked, not taken
# on trust. A failure nobody has written up does not belong here.
KNOWN_FAILED = {
    ("groot_n17", "vis_compile"):
        "GR00T ladder not cumulative; reported STATUS.md:214, E4_PARTS_ABC_GUIDE.md:284",
}
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
    known = sorted({r for r in set(bad) if (name, r) in KNOWN_FAILED})
    new_bad = sorted(set(bad) - set(known))
    print(f"  {name:10s} measured={len(ok):2d}  n/a={len(na)}  FAILED={len(new_bad)}"
          f"  KNOWN={len(known)}  MISSING={len(missing)}"
          + (f"  -> {new_bad}" if new_bad else "")
          + (f"  known:{known}" if known else "")
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
# A directory whose measurement has since been taken SOMEWHERE ELSE is not an outstanding
# defect, and reporting it as BROKEN every ten minutes is the crying-wolf failure this file
# has now hit four times. The entry names its replacement so the claim can be checked, and
# the replacement must actually exist -- a superseded-by that is itself missing is reported
# as a normal breakage rather than quietly forgiven.
SUPERSEDED = {
    "20260922_E5_pi0_reduce-overhead_N4": "20260924_E5_pi0_best_N4",
}

for tag, pat, need, target in (
        ("E11 hybrid",  "results/*_E11*_*",  "model_inputs.csv",       4),
        ("E5 timeline", "results/*_E5_*",    "timeline_summary.csv",  12),
        ("E5b instr",   "results/*_E5b_*",   "stall_attribution.csv",  6),
        # E12 (T014). corun_ratios.csv is the file that carries s, p and r -- a directory
        # with timings but no ratios has not answered the question the experiment exists
        # for. Target 5 per docs/E12_corun/README.md §10 (corun N10, mechanisms, N4,
        # energy, and MODE_30W which waits for the owner's reboot).
        ("E12 corun",   "results/*_E12_*",   "corun_ratios.csv",       5)):
    rows = [(d, (Path(d) / need).exists() and (Path(d) / need).stat().st_size > 0)
            for d in sorted(glob.glob(pat))]
    ok = sum(1 for _, o in rows if o)
    live = [os.path.basename(d) for d, o in rows if not o and _is_live(d)]
    broken_all = [os.path.basename(d) for d, o in rows if not o and not _is_live(d)]

    # E5b's need file, stall_attribution.csv, is written by ONE analyse pass at the end of
    # the whole queue (bench/queues/queue_e5b.sh line 89), not per capture. So a directory
    # whose ncu captures are finished sits without it while the queue works on the NEXT
    # model, and _is_live() cannot see that -- nothing is writing to it. At 21:22 this
    # reported BROKEN: ['20260924_E5b_pi0_eager'] when that directory held three completed
    # captures and was simply waiting its turn to be analysed.
    #
    # Same live-awareness class as the stale JOBS list and the buffered zero-byte stdout:
    # the check was not wrong about the file, it was wrong about what its absence meant.
    # Phases whose `need` file is written by ONE final pass at the end of a queue, not per
    # directory. Such a directory legitimately sits without it while the queue works on the
    # next thing, and _is_live() cannot see that because nothing is writing to it.
    #
    # A TABLE, not a chain of `if tag == ...`. E5b needed this on 2026-09-25 and E12 needed
    # the identical thing an hour later; adding a second special case is how the JOBS
    # allowlist went stale twice. Entries: tag -> (queue log, done marker, partial evidence).
    LATE_NEED = {
        "E5b instr": ("scratch/queues/queue_e5b.log", "E5B_DONE",
                      "results/{d}/ncu/*_details.csv"),
        # e12_2proc_ratios.py writes corun_ratios.csv after the whole matrix. Evidence is
        # raw/*.log, which each role opens IMMEDIATELY -- the per-role JSON is written only
        # when that role's 60 s window ends, so keying on it left the directory looking
        # broken for the first several minutes of a run that was working fine.
        "E12 corun": ("scratch/queues/queue_e12_2proc.log", "E12_2PROC_DONE",
                      "results/{d}/raw/*.log"),
    }
    awaiting = []
    ent = LATE_NEED.get(tag)
    if ent:
        log, marker, partial = ent
        queue_done = any(marker in open(f, errors="ignore").read()
                         for f in glob.glob(log))
        if not queue_done:
            still = []
            for b in broken_all:
                if glob.glob(partial.format(d=b)):
                    awaiting.append(b)
                else:
                    still.append(b)
            broken_all = still

    superseded, broken = [], []
    for b in broken_all:
        rep = SUPERSEDED.get(b)
        if rep and (Path("results") / rep / need).exists():
            superseded.append(f"{b}->{rep}")
        else:
            broken.append(b)
    print(f"  {tag:11s} {ok} usable of {len(rows)} present (target ~{target})"
          + (f"   RUNNING: {live}" if live else "")
          + (f"   BROKEN: {broken[:3]}" if broken else "")
          + (f"   superseded: {superseded}" if superseded else "")
          + (f"   awaiting the final analyse pass: {awaiting}" if awaiting else ""))

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
# Same rule as KNOWN_FAILED in section 1a: a gap that is explained and already queued is
# shown without the alarm marker, so "***" keeps meaning "new". The abandoned compiled E5
# capture is the whole reason bench/queues/queue_e5.sh exists -- its header documents why
# it looked impossible and why a --rung makes it measurable -- so raising it every ten
# minutes until E5 runs tells me nothing I have not already written down and scheduled.
KNOWN_EMPTY = {"20260922_E5_pi0_reduce-overhead_N4":
               "abandoned compiled capture; Phase 3 re-measures it via "
               "bench/queues/queue_e5.sh (--rung, not --compile-mode)"}
known_empty = [d for d in empty if d in KNOWN_EMPTY]
empty = [d for d in empty if d not in KNOWN_EMPTY]
if known_empty:
    print(f"  {len(known_empty)} zero-byte stdout, known and queued: {known_empty}")
if empty:
    print(f"  *** {len(empty)} ZERO-BYTE stdout (a run that wrote nothing): {empty[:3]}")
if empty_live:
    print(f"  ({len(empty_live)} empty stdout in a RUNNING job — output still buffered)")
PYB
echo
echo "--- 1c. CONSOLIDATION COMPLETENESS (CHECK_OK does not cover this)"
$HOME/venvs/pi0/bin/python - <<'PYC'
# check_model_inputs.py validates the SCHEMA of the rows present. It cannot know a part is
# absent. On 2026-09-24 at 17:23 the smolvla consolidation ran with three parts missing and
# the queue printed "EXIT=0 smolvla consolidate" then "CHECK_OK smolvla" -- the MISSING
# lines were there, three lines above, but the pass/fail signal said success. That file is
# what the research side reads DIRECTLY to build the PIM model, so its completeness needs
# to be in the audit and not only in a log nobody re-reads.
import glob, json, os
for d in sorted(glob.glob("results/*_E11_*_inputs")):
    mp = os.path.join(d, "meta.json")
    if not os.path.exists(mp):
        print(f"  {os.path.basename(d):34s} NO meta.json"); continue
    m = json.load(open(mp))
    miss, rows = m.get("parts_missing"), m.get("total_rows")
    if miss:
        print(f"  {os.path.basename(d):34s} rows={rows}  *** INCOMPLETE "
              f"parts_missing={miss} ***")
    else:
        print(f"  {os.path.basename(d):34s} rows={rows}  complete")
PYC
echo
echo "--- 1d. DOES EACH E5 CSV CONTAIN THE METRIC IT EXISTS FOR"
$HOME/venvs/pi0/bin/python - <<'PYR'
# The plan's audit spec asks for "a CSV without the metric it exists for", and section 1b
# does not answer it: it counts a run usable when timeline_summary.csv exists and has rows.
# At 20:01 it reported "E5 timeline 14 usable of 15" while SIX of those fourteen had no
# ae_step row at all -- the GR00T captures carry ae_total instead, because GR00T's action
# head is invoked as one unit and has no per-step denoise_step range. That is a real
# structural difference and not a failure, but a cross-model ae_step table built from that
# directory listing would silently have omitted every DiT model.
#
# So report WHICH region each E5 run actually has. An ae_step run and an ae_total run are
# both fine; a run with neither is broken, and a table must not mix them without saying so.
import csv, glob, os
step, total, neither = [], [], []
for d in sorted(glob.glob("results/*_E5_*")):
    f = os.path.join(d, "timeline_summary.csv")
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        continue
    regs = {x.get("region") for x in csv.DictReader(open(f))}
    n = os.path.basename(d)
    (step if "ae_step" in regs else total if "ae_total" in regs else neither).append(n)
print(f"  ae_step region : {len(step)}  {step}")
print(f"  ae_total only  : {len(total)}  {total}")
if neither:
    print(f"  *** NEITHER ae_step NOR ae_total: {neither} -- these carry no expert region "
          f"and cannot answer what E5 exists to answer ***")
else:
    print("  every E5 run carries an expert region (ae_step or ae_total)")
PYR
echo
echo "--- 1e. ACCEPTANCE COVERAGE: does every task's OWN named deliverable exist?"
# Added 2026-09-25 after E6 was found undelivered for all six models while T004 -- titled
# "E5, E6, E7" -- was marked `done` with a status block that never mentions E6. Every audit
# section above checks experiments I named by hand (E5, E5b, E7, E11, E12), so an experiment
# absent from those lists could not be missed loudly: the checker's scope excluded it. This
# section takes its scope from the TASK FILES instead, so a deliverable nobody remembered is
# still reported. Same failure family as the §2c glob that saw 7 of 33 missing meta.json.
python3 - <<'PYA'
import glob, os, re
from pathlib import Path

rows = []
unexpandable = []
for tf in sorted(glob.glob("tasks/T0*.md")):
    txt = Path(tf).read_text(errors="replace")
    # The named deliverables are the backticked reports/*.md paths in the task body.
    want = sorted(set(re.findall(r"`(reports/[A-Za-z0-9_.+-]+\.(?:md|csv))`", txt)))
    # Some tasks name deliverables with a placeholder -- T008 writes
    # `reports/E5_<m>_timeline.md` for m in {pi05, smolvla}. A literal-path check silently
    # skips those, which would be this section committing the very error it exists to catch.
    # Expand the placeholder over the models the task names; if it cannot be expanded, say so
    # rather than drop it.
    tmpl = sorted(set(re.findall(r"`(reports/[A-Za-z0-9_.+<>-]*<[a-z]+>[A-Za-z0-9_.+-]*\.(?:md|csv))`", txt)))
    models = re.findall(r"`(pi0|pi05|smolvla|groot_n1[567])`", txt)
    seen, order = set(), []
    for m in models:
        if m not in seen:
            seen.add(m); order.append(m)
    for t in tmpl:
        if order:
            want += [re.sub(r"<[a-z]+>", m, t) for m in order]
        else:
            unexpandable.append((os.path.basename(tf), t))
    want = sorted(set(want))
    if not want:
        continue
    status = [ln for ln in txt.splitlines() if ln.startswith("status:")]
    last = status[-1][:60] if status else "NO STATUS BLOCK"
    missing = [w for w in want if not os.path.exists(w)]
    rows.append((os.path.basename(tf), last, want, missing))

bad = [r for r in rows if r[3]]
for name, last, want, missing in rows:
    mark = "***" if missing else "   "
    print(f"  {mark} {name:38s} {len(want)-len(missing)}/{len(want)} present   {last}")
    for m in missing:
        print(f"        MISSING {m}")
if bad:
    closed = [r for r in bad if r[1].startswith("status: done")]
    if closed:
        print(f"  *** {len(closed)} task(s) marked DONE with a named deliverable MISSING -- a "
              f"green label over incomplete work: {[r[0] for r in closed]}")
    else:
        print(f"  ({len(bad)} task(s) have missing deliverables; none of them is marked done)")
else:
    print("  every task's named deliverables exist")
if unexpandable:
    print(f"  *** templated deliverable names this check could NOT expand, so they are NOT "
          f"verified: {unexpandable}")
PYA
echo
echo "--- 2. RUNNING NOW (from /proc; pgrep -f matches itself)"
$HOME/venvs/pi0/bin/python - <<'PYC'
import os
from pathlib import Path

# Match the EXACT argv element, never a substring of the joined cmdline. Substring
# matching has cost this project three incidents -- the shell that creates a heredoc
# contains the pattern, so the matcher finds itself; once it killed its own shell.
# Any measuring script under bench/, discovered rather than listed.
#
# This was a hardcoded allowlist of filenames, in TWO copies that could drift. It went
# stale twice for the same reason: on 2026-09-24 it reported "NOTHING RUNNING" with
# pipelining.py live, I added the E11 names, and on 2026-09-25 it did it again with
# corun.py live. Adding a name fixes the instance; the allowlist IS the defect, because
# every new measuring script is invisible until someone remembers to register it -- and a
# false "NOTHING RUNNING" is worse than crying wolf, since it hides a stalled queue.
#
# Now: a process counts as a job if any argv element is a .py file under this repo's
# bench/ directory. bw_hog.py is excluded where it matters (it is a helper the load
# generators spawn, handled separately below), and progress_monitor.py is a daemon, not
# work. Nothing to keep in sync and no new script to remember.
# REPO is not defined inside these heredocs -- only os and Path are imported -- so this
# resolves bench/ from the cwd the audit is invoked in, which is always the repo root.
_BENCH_DIR = Path("bench").resolve()
_NOT_JOBS = {"bw_hog.py", "progress_monitor.py", "watchdog.py"}


def _is_job_arg(a: str) -> bool:
    """True if this argv element is a measuring script in bench/.

    Basename lookup, so it matches both `bench/corun.py` and an absolute path. A script
    elsewhere sharing a bench/ basename would count -- deliberately conservative, because a
    false "NOTHING RUNNING" hides a stalled queue and a false positive only prints a line.
    """
    if not a.endswith(".py"):
        return False
    base = a.rsplit("/", 1)[-1]
    if base in _NOT_JOBS:
        return False
    return (_BENCH_DIR / base).exists()

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
    named = any(_is_job_arg(a) for a in argv[1:])
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

# Any measuring script under bench/, discovered rather than listed.
#
# This was a hardcoded allowlist of filenames, in TWO copies that could drift. It went
# stale twice for the same reason: on 2026-09-24 it reported "NOTHING RUNNING" with
# pipelining.py live, I added the E11 names, and on 2026-09-25 it did it again with
# corun.py live. Adding a name fixes the instance; the allowlist IS the defect, because
# every new measuring script is invisible until someone remembers to register it -- and a
# false "NOTHING RUNNING" is worse than crying wolf, since it hides a stalled queue.
#
# Now: a process counts as a job if any argv element is a .py file under this repo's
# bench/ directory. bw_hog.py is excluded where it matters (it is a helper the load
# generators spawn, handled separately below), and progress_monitor.py is a daemon, not
# work. Nothing to keep in sync and no new script to remember.
# REPO is not defined inside these heredocs -- only os and Path are imported -- so this
# resolves bench/ from the cwd the audit is invoked in, which is always the repo root.
_BENCH_DIR = Path("bench").resolve()
_NOT_JOBS = {"bw_hog.py", "progress_monitor.py", "watchdog.py"}


def _is_job_arg(a: str) -> bool:
    """True if this argv element is a measuring script in bench/.

    Basename lookup, so it matches both `bench/corun.py` and an absolute path. A script
    elsewhere sharing a bench/ basename would count -- deliberately conservative, because a
    false "NOTHING RUNNING" hides a stalled queue and a false positive only prints a line.
    """
    if not a.endswith(".py"):
        return False
    base = a.rsplit("/", 1)[-1]
    if base in _NOT_JOBS:
        return False
    return (_BENCH_DIR / base).exists()
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
    if any(_is_job_arg(a) for a in argv[1:]):
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

# Scope is EVERY result directory. It used to glob only *_E7_*, *_E11* and *_E5*, which
# meant E4, E8q and E12 runs were never examined: the audit reported 7 gaps where there
# were 33. A completeness check that silently excludes most of the tree reports green over
# a broken thing, which is the failure mode this whole script exists to catch.
missing = []
for d in sorted(glob.glob("results/*")):
    if not os.path.isdir(d) or Path(d, "meta.json").exists():
        continue
    if any(d == v or d.startswith(v) for v in live_dirs):
        continue                       # in progress: meta.json is written at the end
    missing.append(os.path.basename(d))
if missing:
    # Print ALL of them, grouped by experiment. The old code printed `missing[:4]` beside
    # the true count, so an audit reading "7 ... [4 items]" looked self-contradictory.
    byexp = {}
    for m in missing:
        parts = m.split("_")
        byexp.setdefault(parts[1] if len(parts) > 1 else "?", []).append(m)
    print(f"  {len(missing)} finished run(s) with NO meta.json (protocol §5), all of them:")
    for exp in sorted(byexp):
        print(f"      {exp:6s} {len(byexp[exp]):2d}  {', '.join(sorted(byexp[exp]))}")
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
  # Name exit 137. It is 128+9, i.e. SIGKILL, and on this device that is almost always the
  # kernel OOM-killer: Orin shares 62 GB between CPU and GPU and one pi0 process is ~15 GB.
  # A SIGKILLed python writes NO traceback, so its log is 0 bytes and the failure looks
  # identical to a lost child or a broken pipe. That cost real time twice tonight -- at
  # 02:42 I attributed an OOM kill to `nohup ... &` losing a child, wrongly. Saying it here
  # points at dmesg instead of at a theory.
  if printf "%s\n" "$out" | grep -q "EXIT=137"; then
    echo "      ^ EXIT=137 is SIGKILL (128+9). Check the kernel first, not the harness:"
    echo "        sudo dmesg -T | grep -i oom-kill"
    sudo -n /bin/sh -c 'dmesg -T 2>/dev/null | grep -i "Out of memory: Killed" | tail -2' \
      2>/dev/null | sed 's/^/        /'
  fi
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
      2. E11 hybrid inputs T011/T012      DONE 2026-09-24 18:58, all SIX models, every
                                            consolidated file parts_missing=[]:
                                            pi0 1564, pi0.5 1159, SmolVLA 860 rows, and
                                            E11a DiT segments for GR00T n1.7/n1.6/n1.5.
                                            reports/E11_models.md, commit 493c548.
      2b/2c. the fixup, groot_n15 re-runs and the pi0.5 meta.json back-fill are all DONE.
      3. E5 nsys timeline, six models          DONE 2026-09-24, 14/14. The compiled half
                                                 exists: --rung keeps the NVTX regions.
                                                 reports/E5_timeline.md
      4. E5b / T007 instruction level            DONE 2026-09-25, 6/6 configurations, 24
                                                 captures. reports/E5b_summary.md. NO
                                                 configuration meets §7's 60 % memory-
                                                 latency-bound threshold (19.5-47.6 %
                                                 time-weighted).
      5. check_report_numbers.py for E4d/E4g/E4i DONE, 39/39 values verified.
      6. E7 at MODE_50W, unattended            DONE 2026-09-25 00:50, 18 runs, 0 failures.
                                                 Device restored to MAXN and the gate is
                                                 byte-identical to the pre-switch record.
                                                 energy-delay WORSE at 50W in all 12
                                                 pairable points (1.196-1.400x).
      7. T014 / E12 co-run  <- RUNNING. prefix(k) || expert(k-1) on two streams, one GPU.
         The decisive number is s = B_co/B_iso against r = A_iso/B_iso = 1.50.
         Phase 0 DONE: bench/fixtures/e12, five real episode-0 frames, so the fresh-KV
         check has power (one observation would make a stale-KV bug invisible).
         Next: bench/corun.py harness -> equivalence (eager + compiled, with a negative
         control that MUST fail) -> Step 1 timings -> nsys -> mechanisms -> energy.
         NOTHING E12 reports is valid until the equivalence check passes: non-negotiable 1.
      8. LAST, needs a human reboot: E7 at MODE_30W and MODE_15W, and E12 Step 4 rides the
         MODE_30W session (no extra reboot). Never confirm the prompt, never use
         nvpmodel --force. The handoff and exact commands are at the top of STATUS.md and
         bench/queues/queue_e7_capped.sh is written and tested.
      Also open, unclaimed: T006 (E8 LIBERO closed loop -- no accuracy validation exists
         yet for pi0), T008/T009 (floor.csv, E5 ncu per region, E6 roofline; both still
         at `claimed` and part-done under other tasks -- they need auditing against their
         own acceptance criteria).
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
