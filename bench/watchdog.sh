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

# A rung that leaves NO row is invisible to a per-file scan: GR00T's two backbone rungs
# crashed without writing anything and this audit called the model "4 measured, 0 failed".
# So the expected set is named here and anything absent is reported as MISSING.
LEROBOT = ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
           "vis_graphs", "vlm_compile", "vlm_graphs", "whole_graphs", "vlm_compile_nvg",
           "bf16_base", "bf16_vision", "bf16_norms", "bf16_both"]
EXPECTED = {
    "pi0": LEROBOT,
    "pi05": LEROBOT,
    "smolvla": LEROBOT + ["vis_static_pos"],
    "groot_n17": ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
                  "vlm_compile", "vlm_graphs"],
    # N1.6 measured a full ladder at 09:35 and this audit never mentioned it, because the
    # model list below was written before N1.6 existed. An audit only sees what it is told
    # to look for.
    "groot_n16": ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
                  "vlm_compile", "vlm_graphs"],
    "groot_n15": ["eager_fp32", "eager_tf32", "ae_compile", "ae_graphs", "vis_compile",
                  "vlm_compile", "vlm_graphs"],
}
for name, pat in (("pi0",    "results/20260922_E4c_pi0_opt_ladder/raw/rows/*.json"),
                  ("pi05",   "results/20260922_E4c_pi05_opt_ladder/raw/rows/*.json"),
                  ("smolvla","results/20260922_E4c_smolvla_opt_ladder/raw/rows/*.json"),
                  ("groot_n17","results/20260923_E4d_groot_n17_ladder/raw/rows/*.json"),
                  ("groot_n16","results/20260923_E4d_groot_n16_ladder/raw/rows/*.json"),
                  ("groot_n15","results/20260923_E4d_groot_n15_ladder/raw/rows/*.json")):
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
    present = set(ok) | set(na) | set(bad)
    missing = [r for r in EXPECTED.get(name, []) if r not in present]
    print(f"  {name:10s} measured={len(ok):2d}  n/a={len(na)}  FAILED={len(bad)}"
          f"  MISSING={len(missing)}"
          + (f"  -> {sorted(set(bad))}" if bad else ""))
    if na:
        print(f"             n/a: {sorted(set(na))}")
    if missing:
        print(f"             MISSING (no row at all): {missing}")
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
echo
echo "--- 2b. PROFILER CONCURRENCY (owner, 2026-09-23: never two at once)"
$HOME/venvs/pi0/bin/python - <<'PYEOF'
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
    is_ncu = argv[0].rsplit(b'/', 1)[-1] == b'ncu'
    is_target = any(a.endswith(b'ncu_target.py') or a.endswith(b'groot_phases.py')
                    for a in argv[1:])
    if is_ncu or is_target:
        live.append((d.name, is_ncu,
                     b' '.join(argv[:3]).decode('utf-8', 'replace')[:80]))
n_ncu = sum(1 for _, is_ncu, _ in live if is_ncu)
print(f"  profiler processes: {len(live)}  (ncu binaries: {n_ncu})")
for pid, _, c in live:
    print(f"    {pid}: {c}")
if n_ncu > 1:
    print("  *** TWO OR MORE PROFILERS LIVE - STOP ONE NOW "
          "(they are root-owned; plain kill returns Operation not permitted) ***")
elif n_ncu == 0 and live:
    print("  (python targets without an ncu: a capture that has finished profiling)")
PYEOF

echo "--- 3. QUEUE MARKERS"
# glob every queue log: naming two of them by hand meant the 07:22 audit printed markers
# from a queue that had been superseded two queues earlier.
ls -t scratch/T004/queue_*.log 2>/dev/null | head -3 | while read -r f; do
  echo "  [$f]"
  grep -hE "_START |_DONE |^####|EXIT=[^0]" "$f" 2>/dev/null | tail -6 | sed 's/^/    /'
done
echo
echo "--- 4. ORDER: Part A+B all models error-free -> then Part C. Re-derive before acting."
