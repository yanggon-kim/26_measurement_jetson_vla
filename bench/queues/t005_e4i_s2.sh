#!/usr/bin/env bash
# T005 E4i stage 2 (resumed 2026-10-23): GEMM autotuning per part (stage 1: autotuning helps the experts but slows vision /
# prefix), the bf16_islands variants (remaining fp32 Linears to bf16), GR00T re-measures with fallback_random.
# Rows in results/<D>_E4i_<model>_best_search_s2/. Usage: bash bench/queues/t005_e4i_s2.sh <YYYYMMDD> [models...]
set -u
D=${1:?date}; shift
MODELS=${*:-"pi0 pi05 smolvla groot_n17 groot_n16 groot_n15"}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
# GR00T: all_compile_autotune re-measured because its ladder row ran without fallback_random (equivalence not comparable);
# linear_mpad slowed GR00T in stage 1, so one padded variant is kept for the record
GR="all_compile_autotune,ae_at,all_compile_ae_at,all_compile_ae_vis_at,vlm_graphs_ae_at,llm_compile_ae_at,all_compile_ae_at_mpad"
for m in $MODELS; do
  echo "##### $(date -u +%H:%M:%S) $m E4i stage 2"
  case $m in
    pi0) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --out results/${D}_E4i_pi0_best_search_s2 \
           --rungs "bf16_mpad_graphs_ae_at,bf16_mpad_graphs_ae_vlm_at,bf16_mpad_graphs_ae_vis_at,bf16_mpad_graphs_ae_at_islands,bf16_mpad_graphs_autotune_islands" ;;
    pi05) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi05 --out results/${D}_E4i_pi05_best_search_s2 \
           --rungs "bf16_graphs_islands,bf16_graphs_ae_at_islands,bf16_mpad_graphs_ae_at,bf16_mpad_graphs_ae_at_islands,bf16_mpad_graphs_ae_vlm_at_islands" ;;
    smolvla) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model smolvla --out results/${D}_E4i_smolvla_best_search_s2 \
           --rungs "bf16_mpad_graphs_nofuse_islands,bf16_mpad_graphs_islands,bf16_mpad_graphs_ae_at,bf16_mpad_graphs_ae_at_islands" ;;
    groot_n17) ~/venvs/groot17/bin/python bench/opt_ladder.py --model groot_n17 --out "$R/results/${D}_E4i_groot_n17_best_search_s2" \
           --rungs "ae_at,llm_compile_ae_at,llm_graphs_ae_at,ae_at_mpad" ;;
    groot_n16) ~/venvs/groot16/bin/python bench/opt_ladder.py --model groot_n16 --out "$R/results/${D}_E4i_groot_n16_best_search_s2" --rungs "$GR" ;;
    groot_n15) ~/venvs/groot15/bin/python bench/opt_ladder.py --model groot_n15 --out "$R/results/${D}_E4i_groot_n15_best_search_s2" \
           --rungs "$GR,vlm_graphs_mpad_autotune,llm_graphs_mpad_autotune" ;;
  esac
done
echo "##### $(date -u +%H:%M:%S) E4i stage 2 done"
