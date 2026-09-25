#!/usr/bin/env bash
# groot_n15 E11a re-run: the one GR00T version whose reconstruction needed the model's own
# autocast context (adb6e77).
#
# It died at 17:25 with
#     RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
#     at attn.to_k(encoder_hidden_states)     bench/segments_groot.py:139
# because gr00t/model/policy.py:190 wraps n1.5's inference in
# torch.autocast(device_type="cuda", dtype=bfloat16) and the reconstruction did not.
# n1.7 and n1.6 have no autocast in policy.py, which is why only n1.5 failed.
#
# A THIRD script rather than an edit, for the same reason queue_e11_fixup.sh was a second
# one: queue_e11_fixup.sh is executing, and bash reads a running script incrementally from
# a byte offset -- editing one is how preflight pass 7 died with "syntax error near
# unexpected token 'done'".
#
# CHECK_FAILED for groot_n15 is EXPECTED and is not this script failing: the research
# side's check_model_inputs.py MODELS set is {pi0, pi05, smolvla, groot_n17, none}, with no
# groot_n15 or groot_n16. groot_segments.py detects that, says so loudly, records it in
# meta.json and writes the rows anyway. Raised in STATUS.md under "Questions for the
# research side"; it is a schema question, not a defect, and must not be worked around by
# relabelling rows.
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

# One GPU job at a time. Bracketed first character so the pattern does not match the sh
# that runs it -- pkill -f "TreeLauncherSubreaper" once killed the very shell issuing it.
for q in "[q]ueue_e11.sh" "[q]ueue_e11_fixup.sh"; do
  if pgrep -u "$USER" -f "$q" >/dev/null 2>&1; then
    echo "REFUSING: ${q//[\[\]]/} is still running; one GPU job at a time"; exit 2
  fi
done

echo "E11N15_START $(date -Is)"
grep -q "model_autocast_ctx" bench/segments_groot.py || {
  echo "REFUSING: the autocast fix (adb6e77) is not present; the re-run would fail the same way"
  exit 3; }

D="results/${DATE}_E11a_groot_n15_hostseg"
if have "$D"; then
  echo "already complete: $D"
else
  # The failed attempt left a directory with no model_inputs.csv. Protocol §5 forbids
  # overwriting a RESULT directory, but this one holds no result -- it holds a traceback.
  # Move it aside rather than delete it, so the failure stays inspectable.
  if [ -d "$D" ]; then
    mv "$D" "scratch/failed_${DATE}_E11a_groot_n15_hostseg_1725" 2>/dev/null \
      && echo "moved the failed attempt to scratch/failed_${DATE}_E11a_groot_n15_hostseg_1725"
  fi
  echo "########## $(date +%H:%M) groot_n15 E11a DiT block segments (autocast fix)"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot16/bin/python bench/groot_segments.py \
             --outdir $D --version n15 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
             --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
             --data-config oxe_droid"
  echo "EXIT=$? groot_n15 E11a"
  $PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv" \
    && echo "CHECK_OK groot_n15" \
    || echo "CHECK_FAILED groot_n15 (EXPECTED: no groot_n15 in the checker's MODELS set)"
fi

echo "E11N15_DONE $(date -Is)"
