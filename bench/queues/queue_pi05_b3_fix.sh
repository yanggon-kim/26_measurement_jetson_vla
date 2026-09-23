#!/usr/bin/env bash
# pi0.5's B3, with the conditioning-path cast the dtype probe proved is required.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
echo "PI05_B3FIX_START $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }
run $PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_norms,bf16_both
echo "PI05_B3FIX_DONE $(date -Is)"
