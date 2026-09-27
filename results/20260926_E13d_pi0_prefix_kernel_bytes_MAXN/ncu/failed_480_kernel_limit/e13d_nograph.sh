HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec /usr/local/cuda/bin/ncu --profile-from-start off --nvtx --nvtx-include "regex:prefix_f.*/"   --graph-profiling node --cache-control none   --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum,launch__grid_size,launch__block_size   --export "results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/ncu/e13d_nograph" --force-overwrite   /home/yanggon/venvs/pi0/bin/python bench/e13d_target.py --mode ncu --frames 2 --vision-nograph
