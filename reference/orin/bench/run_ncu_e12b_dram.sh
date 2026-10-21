#!/usr/bin/env bash
# E12b follow-up (analysis/E12b_review.md, "owed questions"): the EMC-vs-bytes gap.
#
# The review found the prefix's EMC-calibrated draw (84.6 GB/s) 3.6x above what E6's
# matched-rung bytes imply (~24 GB/s), and the expert's 1.28x. The EMC calibration maps
# controller busy % onto a STREAMING copy's GB/s, so either the prefix really moves ~3.6x more
# DRAM traffic than E6 counts, or busy % overstates bytes for its access pattern. This measures
# the bytes directly: DRAM read + write per kernel, every kernel of the `prefix_alone` and
# `expert_alone` NVTX ranges of bench/e12b_nsys.py (5 frames each, graphs_both, whole GPU --
# the same configuration and the same code path as M2/M3).
#
# --cache-control none: L2 is NOT flushed between workloads, so each sees the cache state the
# real run leaves it (the default, `all`, flushes and would overcount DRAM reads).
#
# ORIN HAS NO dram__ COUNTERS (first attempt, 2026-09-25 22:47, kept in scratch/failed_runs:
# ncu silently omitted dram__bytes_* -- `ncu --query-metrics` lists none -- and the per-NODE
# graph replay then died with "LaunchFailed" after ~480 kernels). So DRAM traffic is measured at
# the L2 boundary: reads = L2 sector FILLS from memory (lts__d_sectors_fill_device + _sysmem,
# 32 B each); writes = L2 write sectors (lts__t_sectors_op_write, an UPPER bound on DRAM writes,
# since a line written twice before eviction is written back once). And --graph-profiling graph:
# each CUDA-graph launch is ONE workload (the graphs_both prefix and the expert are one graph
# each), which is the per-stage total wanted and avoids per-node replay.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=${1:?outdir}
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 2; }
mkdir -p "$OUT/ncu" "$OUT/raw"
if pgrep -x ncu >/dev/null || pgrep -x nsys >/dev/null; then
  echo "REFUSING: another profiler is running"; exit 2
fi
PY=$HOME/venvs/pi0/bin/python
R="$OUT/ncu/e12b_dram.sh"
cat > "$R" <<EOR
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) --profile-from-start off --nvtx \
  --nvtx-include "prefix_alone/" --nvtx-include "expert_alone/" \
  --graph-profiling graph --cache-control none \
  --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum \
  --export "$OUT/ncu/e12b_dram" --force-overwrite \
  $PY bench/e12b_nsys.py --config graphs_both
EOR
t0=$SECONDS
sudo -n /bin/sh "$R" > "$OUT/ncu/e12b_dram.stdout" 2>&1; RC=$?
# CACHE_OWNERSHIP_FIX: the profiled process runs as root; hand the compile caches back.
sudo -n chown -R "$(id -u):$(id -g)" "$OUT" /tmp/triton_yanggon /tmp/torchinductor_yanggon \
    "$HOME/.triton" 2>/dev/null
echo "  ncu rc=$RC in $((SECONDS-t0))s"
[ -f "$OUT/ncu/e12b_dram.ncu-rep" ] || { echo "  NO REPORT"; exit 3; }
sudo -n "$(which ncu)" --import "$OUT/ncu/e12b_dram.ncu-rep" --csv --page raw > "$OUT/ncu/e12b_dram_raw.csv" 2>/dev/null
zstd -19 -q -f "$OUT/ncu/e12b_dram.ncu-rep" -o "$OUT/ncu/e12b_dram.ncu-rep.zst" \
    && rm -f "$OUT/ncu/e12b_dram.ncu-rep"
$PY bench/e12b_dram_analyze.py "$OUT"
