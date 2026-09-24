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
# ## GR00T is absent on purpose
#
# T012 needs a GR00T ModelSpec and a check that its 16 DiT blocks expose a patchable
# forward. Neither exists yet, and MR.get("groot_n17") raises by design rather than
# guessing wrap points -- the same mistake as assuming regions.py works for SmolVLA.
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

echo "E11_DONE $(date -Is)"
