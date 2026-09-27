HOME=/home/yanggon
export HOME
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec /usr/local/cuda/bin/ncu --profile-from-start off --nvtx --nvtx-include "regex:g[0-9]+_M.*/" --cache-control none   --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,gpu__time_duration.sum,launch__grid_size   --export results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN/ncu/gemm_sweep --force-overwrite /home/yanggon/venvs/pi0/bin/python bench/e13d_gemm_sweep.py --run --shapes "816,16384,2048;256,4304,1152;816,2048,16384;816,2048,2048"
