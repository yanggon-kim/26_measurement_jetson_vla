#!/usr/bin/env bash
# Recovery: smolvla E11f (whose re-run died on a NameError I introduced) and the
# re-consolidation it blocks.
#
# 18:13 state: queue_e11_fixup.sh finished. pi0 and pi0.5 are complete. smolvla's
# consolidated file is 842 rows with parts_missing=['E11f_smolvla_pipelining'] -- which the
# queue reported as CHECK_OK, because check_model_inputs.py validates the schema of the rows
# present and cannot see an absent part. That is why bench/watchdog.sh section 1c exists.
#
# Why E11f failed at 18:09, after its measurement had entirely succeeded:
#
#   File "bench/pipelining.py", line 379, in main
#     notes.append(capture_note)
#   NameError: name 'notes' is not defined
#
# My own bug. This driver builds meta["notes"] as a literal AFTER the mode loop, so no such
# name exists at that point; the fix collects into `capture_notes` declared before the loop.
# I had checked that edit with ast.parse, which cannot see a NameError -- pyflakes can, and
# reports the file clean now. Everything upstream worked: the eager-capture fallback fired
# exactly as designed ("layer capture empty under reduce-overhead ... re-capturing from a
# second EAGER instance") and rebuilt 3/6 mapping segments.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
have() { [ -s "$1/model_inputs.csv" ]; }
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

for q in "[q]ueue_e11.sh" "[q]ueue_e11_fixup.sh" "[q]ueue_e11_groot_n15.sh"; do
  pgrep -u "$USER" -f "$q" >/dev/null 2>&1 && { echo "REFUSING: another queue is running"; exit 2; }
done

echo "E11REC_START $(date -Is)"
# Refuse if the NameError fix is absent, rather than spending 20 minutes to rediscover it.
grep -q "capture_notes" bench/pipelining.py || {
  echo "REFUSING: pipelining.py has no capture_notes; the NameError is still there"; exit 3; }
$PY -m pyflakes bench/pipelining.py | grep -q "undefined name" && {
  echo "REFUSING: pyflakes reports an undefined name in pipelining.py"; exit 4; }

D="results/${DATE}_E11f_smolvla_pipelining"
if have "$D"; then
  echo "already complete: $D"
else
  # The two failed attempts left nsys/ and raw/ but no result. Move aside, keep inspectable.
  [ -d "$D" ] && mv "$D" "scratch/failed_${DATE}_E11f_smolvla_pipelining_$(date +%H%M)" \
    && echo "moved the failed attempt aside"
  run "smolvla E11f pipelining (NameError fixed)" \
    $PY bench/pipelining.py --model smolvla --outdir "$D" --steps 4 --seconds 60
fi

D="results/${DATE}_E11_smolvla_inputs"
run "smolvla re-consolidate" $PY bench/consolidate_e11.py --model smolvla --outdir "$D"
$PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv" \
  && echo "CHECK_OK smolvla" || echo "CHECK_FAILED smolvla"

# Completeness, which CHECK_OK cannot see (watchdog section 1c has the same check).
$PY - <<'PYE'
import glob, json, os
bad = 0
for d in sorted(glob.glob("results/20260924_E11_*_inputs")):
    m = json.load(open(os.path.join(d, "meta.json")))
    miss = m.get("parts_missing")
    print(f"  {os.path.basename(d):32s} rows={m.get('total_rows')}  "
          + (f"*** parts_missing={miss} ***" if miss else "complete"))
    bad += bool(miss)
print(f"  {'INCOMPLETE=' + str(bad) if bad else 'all consolidated files complete'}")
PYE

echo "E11REC_DONE $(date -Is)"
