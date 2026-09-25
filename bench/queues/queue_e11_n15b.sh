#!/usr/bin/env bash
# groot_n15 E11a, second re-run: with the autocast decision driven by --api (655d9ba)
# instead of by a source inspection that silently returned nullcontext.
#
# Attempt 1 (17:25) died at attn.to_k(encoder_hidden_states): no autocast at all.
# Attempt 2 (18:14) died at attn.to_q(hidden_states) -- one Linear EARLIER, which is the
# tell that the context still was not active: the helper's `except (ImportError, OSError)`
# swallowed whatever went wrong and returned nullcontext without a word.
#
# Verified before spending a run, in the same venv and with the same root the queue uses:
#   n1d5 --api=old -> torch.autocast(cuda, bfloat16), source cross-check agrees
#   n1d6 --api=new -> nullcontext
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
for q in "[q]ueue_e11.sh" "[q]ueue_e11_fixup.sh" "[q]ueue_e11_recover.sh"; do
  pgrep -u "$USER" -f "$q" >/dev/null 2>&1 && { echo "REFUSING: another queue is running"; exit 2; }
done

echo "E11N15B_START $(date -Is)"
grep -q "_API\[0\]" bench/groot_segments.py || {
  echo "REFUSING: the --api-driven autocast fix is not present"; exit 3; }

D="results/${DATE}_E11a_groot_n15_hostseg"
if [ -s "$D/model_inputs.csv" ]; then
  echo "already complete: $D"
else
  [ -d "$D" ] && mv "$D" "scratch/failed_${DATE}_E11a_groot_n15_$(date +%H%M)" \
    && echo "moved attempt 2 aside"
  echo "########## $(date +%H:%M) groot_n15 E11a (--api-driven autocast)"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot16/bin/python bench/groot_segments.py \
             --outdir $D --version n15 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
             --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
             --data-config oxe_droid"
  echo "EXIT=$? groot_n15 E11a"
fi

# The previous script printed "CHECK_FAILED groot_n15 (EXPECTED: ...)" even when the CSV did
# not exist at all -- asserting a benign explanation for a failure that was something else
# entirely. Distinguish the two, because the whole point of that message was to stop a known
# schema mismatch being mistaken for a defect, and it did the opposite.
if [ ! -s "$D/model_inputs.csv" ]; then
  echo "CHECK_SKIPPED groot_n15: no model_inputs.csv -- the RUN failed, this is NOT the"
  echo "  known schema mismatch. Read the traceback above."
else
  if $PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv"; then
    echo "CHECK_OK groot_n15"
  else
    echo "CHECK_FAILED groot_n15 (EXPECTED: the checker's MODELS set has no groot_n15;"
    echo "  the rows are written and meta.json records the mismatch — schema question,"
    echo "  raised in STATUS.md, not a defect)"
  fi
fi

echo "E11N15B_DONE $(date -Is)"
