#!/usr/bin/env bash
# T005 E4i stage 3 (2026-10-24, after the T007 timeline): GR00T N1.7's eager vision and prefix leave the GPU idle 33-39 %
# of their time, CPU-starved (results/20261024_E5_groot_n17_best_bf16_timeline). Its vision could not be compiled
# (flash-attn varlen got a tensor max_seqlen); vis_fa_static (bench/e4i_candidates.py) passes a static upper bound.
# ae_at (the current best_bf16) is re-timed in the same session as the control.
# Usage: bash bench/queues/t005_e4i_s3.sh <YYYYMMDD>
set -u
D=${1:?date}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
echo "##### $(date -u +%H:%M:%S) groot_n17 E4i stage 3"
~/venvs/groot17/bin/python bench/opt_ladder.py --model groot_n17 --out "$R/results/${D}_E4i_groot_n17_best_search_s3" \
  --rungs "ae_at,vis_fa_compile_ae_at,vis_fa_at_ae_at,vis_fa_graphs_ae_at,all_fa_compile_ae_at,vis_fa_compile_llm_graphs_ae_at"
echo "##### $(date -u +%H:%M:%S) E4i stage 3 done"
