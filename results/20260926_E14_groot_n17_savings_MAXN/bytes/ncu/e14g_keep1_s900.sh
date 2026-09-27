HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
TORCHINDUCTOR_COMPILE_THREADS=1
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_COMPILE_THREADS
. /home/yanggon/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec /usr/local/cuda/bin/ncu --profile-from-start off --nvtx --nvtx-include "frame/" \
  --graph-profiling graph --cache-control none --launch-skip 900 --launch-count 450 \
  --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum \
  --export "results/20260926_E14_groot_n17_savings_MAXN/bytes/ncu/e14g_keep1_s900" --force-overwrite \
  /home/yanggon/venvs/groot/bin/python bench/groot_phases.py --outdir results/20260926_E14_groot_n17_savings_MAXN/bytes/raw/run_keep1 --checkpoint nvidia/GR00T-N1.7-3B --version n17 \
    --steps 4 --warmup 20 --timed 1 --tf32 on --rung vlm_compile --head-mode reduce-overhead \
    --llm-mode max-autotune-no-cudagraphs --keep1 --ncu-region frame
