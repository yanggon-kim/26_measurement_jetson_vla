#!/usr/bin/env bash
# T005 step 4: confirmation run of each model's best_bf16 (bench/opt_ladder.py --rung best_bf16: the rung named in
# results/*_E4i_summary/best_bf16.json, its linear_mpad table replayed), with N = 2 and 6 added for the per-step slope.
# An independent repeat in a fresh process. Usage: bash bench/queues/t005_confirm.sh <YYYYMMDD> [models...]
set -u
D=${1:?date}; shift
MODELS=${*:-"pi0 pi05 smolvla groot_n17 groot_n16 groot_n15"}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
for m in $MODELS; do
  echo "##### $(date -u +%H:%M:%S) $m best_bf16 confirmation"
  case $m in
    groot_n1*) v=${m#groot_n}; ~/venvs/groot$v/bin/python bench/opt_ladder.py --model $m --out "$R/results/${D}_E4i_${m}_best_confirm" \
                 --rung best_bf16 --slope-steps 2,6 --slope-rungs all ;;
    *) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model $m --out results/${D}_E4i_${m}_best_confirm --rung best_bf16 \
                 --slope-steps 2,6 --slope-rungs all ;;
  esac
done
echo "##### $(date -u +%H:%M:%S) confirmation done"
