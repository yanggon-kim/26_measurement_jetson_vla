#!/usr/bin/env bash
# E11 preflight: run every part for every model with the smallest parameters that still
# exercise the real code path, before committing hours to the full matrix.
#
# This rule has already paid for itself several times in this campaign: a --ncu-region
# UnboundLocalError caught before it cost 18 captures, a $MODEL unbound variable in a
# guard, dynamo silently falling back to eager, and -- in the E7 matrix an hour ago --
# assert_real_weights running after the rung's bf16 cast. Every one of those would have
# surfaced at the END of a long run.
#
# What is exercised here that has NEVER run before:
#   * host_segments with --model pi05     (the AdaRMS gate path through intermediates)
#   * host_segments with --model smolvla  (segments_smolvla, an entirely new capture)
#   * handover with a SmolVLA plain-dict KV cache (cache_layers' new branch)
#   * prefix_sweep / mempressure / pipelining with a measured P and S rather than 816/51
#   * groot_segments (T012), which has never been executed at all
#
# A failure here is the point. Read the log, fix, re-run; it costs minutes.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
OUT=scratch/preflight
mkdir -p "$OUT"
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

ok=0; bad=0
step() {                      # $1 label  $2.. command
  local label="$1"; shift
  echo "########## $(date +%H:%M) $label"
  if "$@" > "$OUT/pf_$(echo "$label" | tr ' /' '__').log" 2>&1; then
    echo "  OK"; ok=$((ok+1))
  else
    echo "  FAILED -> $OUT/pf_$(echo "$label" | tr ' /' '__').log"
    tail -12 "$OUT/pf_$(echo "$label" | tr ' /' '__').log" | sed 's/^/    /'
    bad=$((bad+1))
  fi
}

echo "E11_PREFLIGHT_START $(date -Is)"

for M in pi05 smolvla; do
  step "$M E11a" $PY bench/host_segments.py --model "$M" \
      --outdir "$OUT/E11a_$M" --steps 4 --timed 20 --warmup 3 --styles eager
  step "$M E11c" $PY bench/handover.py --model "$M" \
      --outdir "$OUT/E11c_$M" --steps 4 --timed 20 --warmup 3
  step "$M E11d" $PY bench/prefix_sweep.py --model "$M" \
      --outdir-prefix "$OUT/E11d_$M" --steps 4 --warmup 2 --timed 3 --modes eager
  step "$M E11e" $PY bench/mempressure.py --model "$M" \
      --outdir-prefix "$OUT/E11e_$M" --steps 4 --seconds 5 --modes eager
  step "$M E11f" $PY bench/pipelining.py --model "$M" \
      --outdir "$OUT/E11f_$M" --steps 4 --seconds 5 --conc-reps 3 --modes eager
done

# pi0's E11f is a RE-run of a part whose committed numbers predate the two-stream fix,
# so its code path is the one already exercised -- but the fix itself has never run.
step "pi0 E11f (two-stream fix)" $PY bench/pipelining.py --model pi0 \
    --outdir "$OUT/E11f_pi0" --steps 4 --seconds 5 --conc-reps 3 --modes eager

# T012: never executed. Only N1.7, because the research side's checker has no
# groot_n16/groot_n15 in its MODELS set (raised in STATUS.md) -- the other two run in the
# real queue, where the schema mismatch is reported rather than silently accepted.
step "groot_n17 E11a" bash -c "source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
    cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
    exec \$HOME/venvs/groot/bin/python bench/groot_segments.py \
      --outdir $OUT/E11a_groot_n17 --version n17 --checkpoint nvidia/GR00T-N1.7-3B \
      --timed 20 --warmup 3 --styles eager"

echo
echo "E11_PREFLIGHT_DONE $(date -Is)  ok=$ok failed=$bad"
[ "$bad" -eq 0 ] || echo "*** DO NOT START THE FULL E11 QUEUE UNTIL THESE ARE FIXED ***"
