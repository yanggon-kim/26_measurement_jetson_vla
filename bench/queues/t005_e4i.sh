#!/usr/bin/env bash
# T005 E4i stage 1: every best_bf16 candidate per model (bench/e4i_candidates.py), one process each, actions saved for
# the equivalence check. Rows land in results/<D>_E4i_<model>_best_search/raw/rows/; existing rows are skipped.
# Usage: bash bench/queues/t005_e4i.sh <YYYYMMDD> [models...]
set -u
D=${1:?date}; shift
MODELS=${*:-"pi0 pi05 smolvla groot_n17 groot_n16 groot_n15"}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
ATTN_COMMON="bf16_mpad_graphs_attn_sdpa_eager,bf16_mpad_graphs_attn_eager_sdpa,bf16_mpad_graphs_attn_sdpa_sdpa,bf16_mpad_nvg_attn_sdpa_sdpa,bf16_mpad_nvg_attn_fa2_eager,bf16_mpad_nvg_attn_fa2_sdpa"
BASE="bf16_mpad_graphs,bf16_mpad_nvg,bf16_mpad_graphs_autotune,bf16_mpad_nvg_autotune,bf16_mpad_graphs_nofuse,bf16_mpad_whole_graphs,bf16_mpad_whole_graphs_autotune"
GR="ae_graphs_mpad,ae_graphs_autotune,ae_graphs_mpad_autotune,llm_compile_autotune,llm_graphs_autotune,llm_graphs_mpad_autotune,vlm_graphs_autotune,vlm_graphs_mpad_autotune,all_compile_autotune_mpad"
for m in $MODELS; do
  echo "##### $(date -u +%H:%M:%S) $m E4i stage 1"
  case $m in
    pi0)  ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --out results/${D}_E4i_pi0_best_search --rungs "$BASE,$ATTN_COMMON" ;;
    pi05) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi05 --out results/${D}_E4i_pi05_best_search \
            --rungs "$BASE,$ATTN_COMMON,bf16_mpad_nvg_attn_fa2_fa2,bf16_mpad_nvg_attn_sdpa_fa2" ;;
    smolvla) ~/venvs/lerobot/bin/python bench/opt_ladder.py --model smolvla --out results/${D}_E4i_smolvla_best_search --rungs "$BASE" ;;
    groot_n17) ~/venvs/groot17/bin/python bench/opt_ladder.py --model groot_n17 --out "$R/results/${D}_E4i_groot_n17_best_search" \
            --rungs "$GR,vlm_compile_scalar,all_compile_autotune_scalar,vlm_graphs_autotune_scalar" ;;
    groot_n16|groot_n15) v=${m#groot_n}; ~/venvs/groot$v/bin/python bench/opt_ladder.py --model $m --out "$R/results/${D}_E4i_${m}_best_search" --rungs "$GR" ;;
  esac
done
echo "##### $(date -u +%H:%M:%S) E4i stage 1 done"
