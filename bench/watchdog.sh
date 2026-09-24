#!/usr/bin/env bash
# Wake the agent after N minutes WITH the audit already rendered, so the wake-up is the
# audit and not a status glance (owner, 2026-09-23): past correctness, what is running,
# what is next. Usage: bash bench/watchdog.sh [minutes]
MIN="${1:-25}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
sleep $((MIN * 60))

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

for tag, pat, need, target in (
        ("E11 hybrid",  "results/*_E11*_*",  "model_inputs.csv",       4),
        ("E5 timeline", "results/*_E5_*",    "timeline_summary.csv",  12),
        ("E5b instr",   "results/*_E5b_*",   "stall_attribution.csv",  6)):
    rows = [(d, (Path(d) / need).exists() and (Path(d) / need).stat().st_size > 0)
            for d in sorted(glob.glob(pat))]
    ok = sum(1 for _, o in rows if o)
    broken = [os.path.basename(d) for d, o in rows if not o]
    print(f"  {tag:11s} {ok} usable of {len(rows)} present (target ~{target})"
          + (f"   BROKEN: {broken[:3]}" if broken else ""))

# A 0-byte stdout is a run that produced nothing while leaving a directory behind.
empty = [f for f in glob.glob("results/*/raw/*stdout*") if os.path.getsize(f) == 0]
if empty:
    names = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in empty[:3]]
    print(f"  *** {len(empty)} ZERO-BYTE stdout (a run that wrote nothing): {names}")
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
        "build_inst_inventory.py", "libero_eval.py")
me, n = os.getpid(), 0
for d in sorted(Path('/proc').iterdir(), key=lambda p: p.name):
    if not d.name.isdigit() or int(d.name) == me:
        continue
    try:
        argv = [a for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
    except OSError:
        continue
    if not argv:
        continue
    exe = argv[0].rsplit(b'/', 1)[-1]
    is_py = exe.startswith(b'python') or exe in (b'ncu', b'nsys')
    hit = any(a.rsplit(b'/', 1)[-1].decode('utf-8', 'replace') in JOBS for a in argv[1:])
    if is_py and (hit or exe in (b'ncu', b'nsys')):
        print(f"  {d.name}: {b' '.join(argv).decode('utf-8', 'replace')[:130]}")
        n += 1
if not n:
    print("  NOTHING RUNNING  <-- if the queue is not done, this is the defect")
elif n > 1:
    print("  *** MORE THAN ONE GPU JOB LIVE -- they are sharing the GPU and every "
          "number from both is confounded ***")
PYC
echo
echo "--- 2b. PROFILER CONCURRENCY (owner, 2026-09-23: never two at once)"
$HOME/venvs/pi0/bin/python - <<'PYD'
import os
from pathlib import Path
me = os.getpid()
live = []
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    try:
        argv = [a for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
    except OSError:
        continue
    if not argv:
        continue
    # exact argv element only; substring-matching the joined cmdline self-matches
    is_ncu = argv[0].rsplit(b'/', 1)[-1] in (b'ncu', b'nsys')
    is_target = any(a.endswith(b'ncu_target.py') or a.endswith(b'nsys_target.py')
                    for a in argv[1:])
    if is_ncu or is_target:
        live.append((d.name, is_ncu,
                     b' '.join(argv[:3]).decode('utf-8', 'replace')[:80]))
n_prof = sum(1 for _, is_ncu, _ in live if is_ncu)
print(f"  profiler processes: {len(live)}  (ncu/nsys binaries: {n_prof})")
for pid, _, c in live:
    print(f"    {pid}: {c}")
if n_prof > 1:
    print("  *** TWO OR MORE PROFILERS LIVE - STOP ONE NOW "
          "(they are root-owned; plain kill returns Operation not permitted) ***")
elif n_prof == 0 and live:
    print("  (python targets without a profiler: a capture that has finished profiling)")
PYD
echo
echo "--- 2c. ORPHANS (root-owned survivors of a killed queue, which load the machine)"
ps -eo pid,user,etime,args --no-headers 2>/dev/null \
  | awk '$3 ~ /bin\/python$/ && $4 ~ /compile_worker/ {n[$2]++} END {for (u in n) printf "  %d orphaned inductor compile_worker owned by %s\n", n[u], u}'
pgrep -x tegrastats >/dev/null 2>&1 && echo "  *** tegrastats is running outside a measurement -- it corrupts the next power log" || true
echo
echo "--- 3. QUEUE MARKERS"
# glob every queue log: naming two of them by hand meant the 07:22 audit printed markers
# from a queue that had been superseded two queues earlier.
ls -t scratch/T004/queue_*.log scratch/queues/*.log 2>/dev/null | head -3 | while read -r f; do
  echo "  [$f]"
  grep -hE "_START |_DONE |^####|EXIT=[^0]" "$f" 2>/dev/null | tail -6 | sed 's/^/    /'
done
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

      1. E7 energy at MAXN, six models, eager + best     <- the empty column; no power
                                                            or energy number exists for
                                                            any model, and PIM is joules
      2. E11 hybrid inputs T011/T012 (pi05, smolvla, groot)  <- the PIM model's direct
                                                            input; needs per-model segment
                                                            code, not a --model flag
      3. E5 nsys timeline, six models (the compiled half + the busy 16 %)
      4. E5b / T007 instruction level, six models
      5. check_report_numbers.py extended to E4d/E4g/E4i  (CPU-only; run between GPU jobs)
      6. E7 at MODE_50W, unattended -- decline any reboot prompt and record it
      7. LAST, needs a human reboot: E7 at MODE_30W and MODE_15W. Never confirm the
         prompt, never use nvpmodel --force; write "next: resume E7 at mode <id>" at the
         top of STATUS.md and hand off with the exact commands.
ORD
