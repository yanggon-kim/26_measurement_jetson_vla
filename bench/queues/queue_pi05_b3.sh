#!/usr/bin/env bash
# After GR00T's ladder: diagnose pi0.5's B3 dtype failure, then re-run the rungs that
# the fix (and the corrected _dominant_dtype) affect.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python

echo "PI05_B3_START $(date -Is)"
while ! grep -q GROOT_QUEUE_DONE scratch/T004/queue_groot.log 2>/dev/null; do sleep 60; done
echo "groot clear $(date -Is)"

run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

# 1. The diagnosis, which decides the fix. Nothing is changed before this prints.
run $PY bench/pi05_dtype_probe.py --model pi05 --prep base
run $PY bench/pi05_dtype_probe.py --model pi05 --prep base,norms

# 2. pi0's bf16_base is re-run because _dominant_dtype was counting parameter TENSORS
#    rather than elements, so pi0 read as fp32 and ran a cast it did not need instead of
#    recording 'not applicable'. The old row is removed so it cannot be mistaken for a
#    measurement of something.
rm -f results/20260922_E4c_pi0_opt_ladder/raw/rows/bf16_base_N10.json
run $PY bench/opt_ladder.py --model pi0 --outdir results/20260922_E4c_pi0_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_base

echo "PI05_B3_DONE $(date -Is)"
