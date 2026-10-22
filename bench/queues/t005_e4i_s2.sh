#!/usr/bin/env bash
# T005 E4i stage 2: GEMM autotuning per part (stage 1 showed autotuning helps the expert but can slow vision / prefix).
# Rows in results/<D>_E4i_<model>_best_search_s2/. Usage: bash bench/queues/t005_e4i_s2.sh <YYYYMMDD> [models...]
set -u
D=${1:?date}; shift
MODELS=${*:-"pi0 pi05 smolvla groot_n17 groot_n16 groot_n15"}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
LR="bf16_mpad_graphs_ae_at,bf16_mpad_graphs_ae_vlm_at,bf16_mpad_graphs_ae_vis_at"
GR="ae_at_mpad,all_compile_ae_at,all_compile_ae_at_mpad,vlm_graphs_ae_at_mpad,all_compile_ae_vis_at_mpad"
for m in $MODELS; do
  echo "##### $(date -u +%H:%M:%S) $m E4i stage 2"
  case $m in
    pi0|pi05|smolvla) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model $m --out results/${D}_E4i_${m}_best_search_s2 --rungs "$LR" ;;
    groot_n17) ~/venvs/groot17/bin/python bench/opt_ladder.py --model groot_n17 --out "$R/results/${D}_E4i_groot_n17_best_search_s2" \
                 --rungs "ae_at,ae_at_mpad,llm_graphs_ae_at,llm_graphs_ae_at_mpad" ;;
    groot_n16|groot_n15) v=${m#groot_n}; ~/venvs/groot$v/bin/python bench/opt_ladder.py --model $m --out "$R/results/${D}_E4i_${m}_best_search_s2" --rungs "$GR" ;;
  esac
done
echo "##### $(date -u +%H:%M:%S) E4i stage 2 done"
