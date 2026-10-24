#!/usr/bin/env bash
# T005 E4i stage 4 (2026-10-25, the owner's decision): kv_shallow on π0 / π0.5's leaders. LeRobot's denoise_step deep-copies
# the prefix KV cache at every step (results/20261025_E4_cpu_overhead_diag: bitwise-equal without it, π0 expert
# 64.7 -> 50.1 ms). The current best_bf16 of each model is re-timed in the same session as the control.
# Usage: bash bench/queues/t005_e4i_s4.sh <YYYYMMDD> [models...]
set -u
D=${1:?date}; shift
MODELS=${*:-"pi0 pi05"}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
for m in $MODELS; do
  echo "##### $(date -u +%H:%M:%S) $m E4i stage 4"
  case $m in
    pi0) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --out results/${D}_E4i_pi0_best_search_s4 \
           --rungs "bf16_mpad_graphs_ae_at,bf16_mpad_graphs_ae_at_kvs,bf16_mpad_graphs_ae_at_islands_kvs" ;;
    pi05) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi05 --out results/${D}_E4i_pi05_best_search_s4 \
           --rungs "bf16_graphs_ae_at_islands,bf16_graphs_ae_at_islands_kvs,bf16_mpad_graphs_ae_at_islands_kvs" ;;
  esac
done
echo "##### $(date -u +%H:%M:%S) E4i stage 4 done"
