#!/usr/bin/env bash
# T016 step 1d (EV-02c): per-kernel DRAM traffic of the pi0 bf16_both rung PREFIX.
# Metrics only -- the E12b set (Orin has no dram__ counters, so reads = L2 fills, writes <= L2 write
# sectors); no --set / --section (those request dram__ counters and killed the first E12b attempt).
# --graph-profiling node first (the rung's embed_image is a CUDA-graph tree); if that fails the caller
# re-runs with FALLBACK=1: embed_image compiled mode="default" (no graph), recorded as such.
# --cache-control none, as E12b. 2 frames, each in NVTX range prefix_f<k>.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=${1:?outdir}; FALLBACK=${FALLBACK:-0}
[ -e "$OUT/ncu/e13d.ncu-rep.zst" ] && { echo "REFUSING: capture exists in $OUT"; exit 2; }
mkdir -p "$OUT/ncu" "$OUT/raw"
if pgrep -x ncu >/dev/null || pgrep -x nsys >/dev/null; then echo "REFUSING: another profiler is running"; exit 2; fi
PY=$HOME/venvs/pi0/bin/python
if [ "$FALLBACK" = 1 ]; then GP=node; EXTRA=--vision-nograph; TAG=nograph; else GP=node; EXTRA=; TAG=rung; fi
R="$OUT/ncu/e13d_${TAG}.sh"
cat > "$R" <<EOR
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) --profile-from-start off --nvtx --nvtx-include "regex:prefix_f.*/" \
  --graph-profiling $GP --cache-control none \
  --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum,launch__grid_size,launch__block_size \
  --export "$OUT/ncu/e13d_${TAG}" --force-overwrite \
  $PY bench/e13d_target.py --mode ncu --frames 2 $EXTRA
EOR
t0=$SECONDS
sudo -n /bin/sh "$R" > "$OUT/ncu/e13d_${TAG}.stdout" 2>&1; RC=$?
sudo -n chown -R "$(id -u):$(id -g)" "$OUT" /tmp/triton_yanggon /tmp/torchinductor_yanggon "$HOME/.triton" 2>/dev/null
echo "  ncu rc=$RC in $((SECONDS-t0))s (tag $TAG)"
[ -f "$OUT/ncu/e13d_${TAG}.ncu-rep" ] || { echo "  NO REPORT"; exit 3; }
grep -q "LaunchFailed\|==ERROR==" "$OUT/ncu/e13d_${TAG}.stdout" && echo "  *** ncu reported an ERROR (see stdout)"
sudo -n "$(which ncu)" --import "$OUT/ncu/e13d_${TAG}.ncu-rep" --csv --page raw > "$OUT/ncu/e13d_${TAG}_raw.csv" 2>/dev/null
zstd -19 -q -f "$OUT/ncu/e13d_${TAG}.ncu-rep" -o "$OUT/ncu/e13d_${TAG}.ncu-rep.zst" && rm -f "$OUT/ncu/e13d_${TAG}.ncu-rep"
exit $RC
