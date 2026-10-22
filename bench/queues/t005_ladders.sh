#!/usr/bin/env bash
# T005 Part B: every ladder at the model's own N (deliverable 1 first), then pi0's N = 4 control.
# One process per rung (bench/opt_ladder.py); each child passes the protocol §1 gate (GPU idle) before loading.
# Usage: bash bench/queues/t005_ladders.sh <YYYYMMDD>
set -u
D=${1:?date}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
LR="eager_fp32,eager_tf32,eager_bf16,bf16_base,ae_compile,ae_graphs,vis_compile,vis_static_pos,vis_graphs,vlm_compile,vlm_compile_nvg,vlm_graphs,whole_graphs,bf16_vision,bf16_norms,bf16_both,bf16_both_graphs"
GR="eager_fp32,eager_tf32,ae_compile,ae_graphs,vis_compile,vis_graphs,vlm_compile,vlm_graphs,all_compile_autotune,llm_compile,llm_graphs"
for m in pi0 pi05 smolvla; do
  echo "##### $(date -u +%H:%M:%S) $m ladder"
  ~/venvs/lerobot/bin/python bench/opt_ladder.py --model $m --out results/${D}_E4c_${m}_opt_ladder --rungs $LR \
      --slope-steps 2,6 --slope-rungs whole_graphs
done
for v in 17 16 15; do
  echo "##### $(date -u +%H:%M:%S) groot_n$v ladder"
  ~/venvs/groot$v/bin/python bench/opt_ladder.py --model groot_n$v --out "$R/results/${D}_E4d_groot_n${v}_ladder" \
      --rungs $GR --slope-steps 2,6 --slope-rungs all
done
echo "##### $(date -u +%H:%M:%S) pi0 N=4 control"
~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --n 4 --out results/${D}_E4c_pi0_opt_ladder --rungs $LR \
    --slope-steps 2,6 --slope-rungs whole_graphs
echo "##### $(date -u +%H:%M:%S) done"
