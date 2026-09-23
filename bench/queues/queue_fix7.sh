#!/usr/bin/env bash
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
echo "FIX7_START $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }
run $PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_norms,bf16_both
echo "FIX7_DONE $(date -Is)"
bash bench/queues/queue_n1d6.sh
