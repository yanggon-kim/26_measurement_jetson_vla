HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRANSFORMERS_OFFLINE=1
TRITON_CACHE_DIR=/home/yanggon/.cache/triton_groot
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
. /home/yanggon/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec /usr/local/cuda/bin/ncu \
  --profile-from-start off --nvtx --nvtx-include "vlm_prefix/" \
  -c 40 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section WarpStateStats --section Occupancy \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio \
  --export "results/20260923_E4g_c1_groot_n17_best/ncu/groot_n17_best_vlm_prefix" --force-overwrite \
  /home/yanggon/venvs/groot/bin/python bench/groot_phases.py \
    --outdir results/20260923_E4g_c1_groot_n17_best --checkpoint nvidia/GR00T-N1.7-3B --version n17 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T \
     --embodiment-tag oxe_droid_relative_eef_relative_joint --warmup 8 --timed 1 --rung ncu_vlm_prefix \
    --ncu-region vlm_prefix --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs
