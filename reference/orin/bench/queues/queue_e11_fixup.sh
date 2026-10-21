#!/usr/bin/env bash
# E11 fixup: the SmolVLA compiled-mode points that queue_e11.sh lost to the
# static-posids bug, plus the re-consolidation they invalidate.
#
# ## What went wrong
#
# smolvla E11d died at 16:24 on 2026-09-24 with cudaErrorStreamCaptureInvalidated.
# Cause and fix are in f7b91a9: prefix_sweep, mempressure and pipelining each called
# MR.load_policy(compile_mode=...) directly and so never got the static-posids prep that
# E7's fix had put in opt_ladder.effective_prep(). The gate now lives in load_policy.
#
# bench/mempressure.py was ALREADY RUNNING when that commit landed, with the unpatched
# module imported, so its reduce-overhead point was expected to fail too and is re-run
# here unconditionally-if-missing rather than assumed good.
#
# ## Why a second script instead of re-running queue_e11.sh
#
# Two reasons, both learned the hard way today.
#
# 1. queue_e11.sh's `have` guard keys on the EAGER directory:
#        have "results/..._E11d_${M}_eager_prefixsweep" || run ...
#    so a run whose eager half succeeded and whose compiled half crashed is
#    indistinguishable from a complete one. Re-running queue_e11.sh would skip exactly
#    the points that are missing. (Fix that guard in queue_e11.sh once nothing is
#    executing it -- recorded in scratch/FIXME_sweep_selfmatch.txt.)
#
# 2. queue_e11.sh was still executing when this was written. bash reads a script
#    incrementally from a byte offset, so editing a running one makes it resume at the
#    wrong place: that is how preflight pass 7 died with "syntax error near unexpected
#    token 'done'" after re-running a step out of order.
#
# Run AFTER E11_DONE appears in scratch/queues/queue_e11.log. Never alongside it -- one
# GPU job at a time.
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
run() {
  local label="$1"; shift
  echo "########## $(date +%H:%M) $label"
  sweep
  "$@"
  echo "EXIT=$? $label"
}

# Refuse to start if the main queue is still going. Matched by exact argv element of the
# right interpreter, not by substring of a joined cmdline -- substring matching has cost
# this project three incidents, once killing its own shell.
if pgrep -u "$USER" -f "[q]ueue_e11.sh" >/dev/null 2>&1; then
  echo "REFUSING: queue_e11.sh is still running; one GPU job at a time"; exit 2
fi

echo "E11FIX_START $(date -Is)"
grep -n "smolvla_static_posids_prep" bench/model_registry.py >/dev/null || {
  echo "REFUSING: the load_policy gate (f7b91a9) is not present; the re-runs would fail the same way"
  exit 3; }

M=smolvla

# Only the compiled halves. The eager directories are complete and are NOT touched:
# re-measuring them would produce a second set of numbers for a point that already has
# one, and invite someone to average them as if independent.
have "results/${DATE}_E11d_${M}_reduce-overhead_prefixsweep" || \
  run "$M E11d prefix sweep (reduce-overhead, re-run)" \
  $PY bench/prefix_sweep.py --model "$M" \
      --outdir-prefix "results/${DATE}_E11d_${M}" --steps 4 --modes reduce-overhead

have "results/${DATE}_E11e_${M}_reduce-overhead_mempressure" || \
  run "$M E11e memory pressure (reduce-overhead, re-run)" \
  $PY bench/mempressure.py --model "$M" \
      --outdir-prefix "results/${DATE}_E11e_${M}" --steps 4 --seconds 60 \
      --modes reduce-overhead

# pipelining writes ONE directory holding both modes, so a compiled-mode crash loses the
# whole run rather than half of it.
D="results/${DATE}_E11f_${M}_pipelining"
have "$D" || run "$M E11f pipelining (re-run)" \
  $PY bench/pipelining.py --model "$M" --outdir "$D" --steps 4 --seconds 60

# Consolidation ran while these were missing, so its output is stale by construction.
D="results/${DATE}_E11_${M}_inputs"
run "$M re-consolidate" $PY bench/consolidate_e11.py --model "$M" --outdir "$D"
$PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv" \
  && echo "CHECK_OK $M" || echo "CHECK_FAILED $M"

# ---- pi0: re-consolidate, because its published table still cites the CONFOUNDED E11f --
#
# results/20260922b_E11_pi0_inputs/meta.json provenance says
#     E11f_pi0_pipelining  <-  20260922b_E11f_pi0_pipelining
# and that run's bench_commit predates the two-stream fix 773d2a9, so its two-stream
# numbers are the GIL-confounded ones T011 said are not reusable. Today's
# 20260924_E11f_pi0_pipelining exists precisely to replace them -- but queue_e11.sh only
# consolidates `for M in pi05 smolvla`, so pi0's consolidated model_inputs.csv was never
# rebuilt and still carries the superseded rows.
#
# That matters more than an ordinary stale file: model_inputs.csv is the one artefact the
# research side reads DIRECTLY to build the PIM model, so leaving it would publish a
# confounded pi0 row next to correct pi0.5 and SmolVLA rows in the same table.
#
# No re-measurement is needed. consolidate_e11.py's pick_run() takes the newest dated
# directory per part, so a fresh run picks 20260924_E11f_pi0_pipelining for E11f and
# leaves every other part on the source it already had. The old consolidated directory is
# kept, not overwritten (protocol §5).
D="results/${DATE}_E11_pi0_inputs"
run "pi0 re-consolidate (picks up the fixed E11f)" \
  $PY bench/consolidate_e11.py --model pi0 --outdir "$D"
$PY docs/reference_scripts/check_model_inputs.py "$D/model_inputs.csv" \
  && echo "CHECK_OK pi0" || echo "CHECK_FAILED pi0"

echo "E11FIX_DONE $(date -Is)"
