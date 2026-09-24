#!/usr/bin/env bash
# E11 hybrid inputs (T011): pi0.5 and SmolVLA, plus pi0's E11f re-run.
#
# E11 produces `model_inputs.csv`, which is the one artefact the research side reads
# DIRECTLY to build its processing-in-memory model: per-segment bytes, FLOPs, measured
# times and the handover costs around them. It exists for pi0 only.
#
# ## The pi0 re-run is not optional
#
# bench/pipelining.py's two-stream measurement was fixed in 773d2a9 (2026-09-22 15:13):
# the first version created a high-priority stream, never used it, and ran the contender
# on a second Python THREAD, so both activities went to the default stream and the 1.9x
# slowdown it reported was largely GIL contention. Both committed E11f runs carry
# bench_commit de325f77 and 5eeddd5c, which PREDATE that fix. So pi0's two-stream numbers
# on disk are still confounded and the corrected ones have never been measured.
#
# ## What is NOT re-run, and why
#
# E11b (bench/xfer_latency.py) measures host<->device copies with no model in them. It is
# kept for pi0, where it already ran as its own part, and dropped for every later model
# per T011 -- re-running it would produce the same numbers under a different name and
# invite someone to average them as independent measurements. The exact transfer figure
# each later model needs comes from its own E11c instead.
#
# ## GR00T gets E11a only, and that is the honest scope
#
# DiT.forward (dit.py:310-326) calls block(...), so transformer_blocks[i].forward IS a
# valid wrap point -- checked before anything was written, because assuming it would have
# been the same mistake as assuming regions.py worked for SmolVLA. So E11a (the per-block
# segment split) runs, through bench/groot_segments.py.
#
# The other E11 parts do NOT run for GR00T, and this is a scope statement rather than a
# gap: E11c/d/e/f are built on the LeRobot phase loops and on a transformers KV cache,
# and GR00T has neither -- its prefix is a whole VLM invoked as one unit and its action
# head takes the backbone output, not a cache. Porting them is a separate piece of work.
# What GR00T contributes here is the per-segment bytes, FLOPs and times of its DiT block,
# which is what the PIM model needs from a DiT architecture.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

# Bracket the first character so the pattern in THIS command line does not match itself.
# `pkill -f "TreeLauncherSubreaper"` kills the very sh that runs it, because that sh's
# own argv contains the string -- it printed "Terminated" and skipped everything after it
# in the E7 queue.
sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}

# "done" means the file exists AND holds rows, not merely that a directory was created.
have() { [ -s "$1/model_inputs.csv" ]; }

run() {                       # $1 label  $2.. command
  local label="$1"; shift
  echo "########## $(date +%H:%M) $label"
  sweep
  "$@"
  echo "EXIT=$? $label"
}

echo "E11_START $(date -Is)"

# ---- pi0: only the part whose committed numbers are confounded ------------------------
D="results/${DATE}_E11f_pi0_pipelining"
have "$D" || run "pi0 E11f (two-stream fix, re-run)" \
  $PY bench/pipelining.py --model pi0 --outdir "$D" --steps 4 --seconds 60

# ---- pi0.5 and SmolVLA: the full set ---------------------------------------------------
for M in pi05 smolvla; do
  # N: pi0.5 and SmolVLA both default to 10, but E11's segment work is per-STEP and the
  # published pi0 set used 4, so 4 keeps the cross-model table comparable.
  D="results/${DATE}_E11a_${M}_hostseg"
  have "$D" || run "$M E11a host segments" \
    $PY bench/host_segments.py --model "$M" --outdir "$D" --steps 4 --timed 1000

  D="results/${DATE}_E11c_${M}_handover"
  have "$D" || run "$M E11c handover" \
    $PY bench/handover.py --model "$M" --outdir "$D" --steps 4 --timed 1000

  have "results/${DATE}_E11d_${M}_eager_prefixsweep" || run "$M E11d prefix sweep" \
    $PY bench/prefix_sweep.py --model "$M" \
        --outdir-prefix "results/${DATE}_E11d_${M}" --steps 4

  have "results/${DATE}_E11e_${M}_eager_mempressure" || run "$M E11e memory pressure" \
    $PY bench/mempressure.py --model "$M" \
        --outdir-prefix "results/${DATE}_E11e_${M}" --steps 4 --seconds 60

  D="results/${DATE}_E11f_${M}_pipelining"
  have "$D" || run "$M E11f pipelining" \
    $PY bench/pipelining.py --model "$M" --outdir "$D" --steps 4 --seconds 60

  D="results/${DATE}_E11_${M}_inputs"
  run "$M consolidate" $PY bench/consolidate_e11.py --model "$M" --outdir "$D"
  # The checker is run IN PLACE, never copied or adapted (protocol §13.8).
  $PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv" \
    && echo "CHECK_OK $M" || echo "CHECK_FAILED $M"
done

# ---- GR00T: E11a only ------------------------------------------------------------------
groot_e11a() {
  local ver="$1" venv="$2" root="$3"; shift 3
  local d="results/${DATE}_E11a_groot_${ver}_hostseg"
  if have "$d"; then
    echo "########## $(date +%H:%M) groot_$ver E11a already done"; return
  fi
  echo "########## $(date +%H:%M) groot_$ver E11a DiT block segments"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_segments.py \
             --outdir $d --version $ver ${root:+--gr00t-root $root} $*"
  echo "EXIT=$?"
  $PY docs/reference_scripts/check_model_inputs.py "$d/model_inputs.csv" \
    && echo "CHECK_OK groot_$ver" || echo "CHECK_FAILED groot_$ver"
}

groot_e11a n17 groot "" --checkpoint nvidia/GR00T-N1.7-3B
groot_e11a n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
groot_e11a n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
      --data-config oxe_droid

echo "E11_DONE $(date -Is)"
