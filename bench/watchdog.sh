#!/usr/bin/env bash
# Wake the agent after N minutes WITH the audit already rendered, so the wake-up is the
# audit and not a status glance (owner, 2026-09-23): past correctness, what is running,
# what is next. Usage: bash bench/watchdog.sh [minutes]
MIN="${1:-25}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
sleep $((MIN * 60))

echo "=================== WATCHDOG AUDIT $(date +%H:%M) ==================="
echo
echo "--- 1. PAST EXECUTION: every Part A/B row, with its status"
$HOME/venvs/pi0/bin/python - <<'PY'
import glob, json, os
for name, pat in (("pi0",    "results/20260922_E4c_pi0_opt_ladder/raw/rows/*.json"),
                  ("pi05",   "results/20260922_E4c_pi05_opt_ladder/raw/rows/*.json"),
                  ("smolvla","results/20260922_E4c_smolvla_opt_ladder/raw/rows/*.json"),
                  ("groot_n17","results/20260923_E4d_groot_n17_ladder/raw/rows/*.json")):
    fs = sorted(glob.glob(pat))
    ok, na, bad = [], [], []
    for f in fs:
        r = json.load(open(f))
        tag = f"{r['rung']}"
        v = r.get("e2e_median_ms")
        # NaN is NOT a measurement. The first version of this audit tested
        # `v not in ("", None)`, and a not-applicable rung writes float('nan'), so it
        # reported SmolVLA as 14 measured / 0 n/a when the truth was 11 / 3. An audit
        # that miscounts is worse than no audit.
        # ...and a number stored as a STRING is still a number: pi0's N=4 rows were
        # written by an older path that stringified it, so a strict isinstance check
        # called 8 good rungs failed. Coerce, then reject NaN and non-positive.
        try:
            fv = float(v)
            measured = fv == fv and fv > 0
        except (TypeError, ValueError):
            measured = False
        if r.get("status") == "not applicable" or r.get("not_applicable_reason"):
            na.append(tag)
        elif measured:
            ok.append(tag)
        else:
            bad.append(tag)
    print(f"  {name:10s} measured={len(ok):2d}  n/a={len(na)}  FAILED={len(bad)}"
          + (f"  -> {sorted(set(bad))}" if bad else ""))
    if na:
        print(f"             n/a: {sorted(set(na))}")
PY
echo
echo "--- 2. RUNNING NOW (from /proc; pgrep -f matches itself)"
$HOME/venvs/pi0/bin/python - <<'PY'
from pathlib import Path
n = 0
for d in sorted(Path('/proc').iterdir()):
    if not d.name.isdigit():
        continue
    try:
        c = (d/'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        continue
    if any(k in c for k in ('opt_ladder', 'groot_phases', 'paper_repro', 'ncu', 'nsys')) \
            and 'grep' not in c:
        print(f"  {d.name}: {c[:120]}"); n += 1
if not n:
    print("  NOTHING RUNNING  <-- if the queues are not done, this is the defect")
PY
echo
echo "--- 3. QUEUE MARKERS"
grep -hE "PARTB_FINISH_START|PARTB_FINISH_DONE|GROOT_QUEUE_START|GROOT_QUEUE_DONE|^####|EXIT=[^0]" \
    scratch/T004/queue_partb_finish.log scratch/T004/queue_groot.log 2>/dev/null | tail -12
echo
echo "--- 4. ORDER: Part A+B all models error-free -> then Part C. Re-derive before acting."
