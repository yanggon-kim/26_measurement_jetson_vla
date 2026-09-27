#!/usr/bin/env bash
# T017 2b: per-variant DRAM bytes, E12b graph-level method, target bench/e14b_padtrim.py --mode ncu (prefix, expert, vision, vlm graphs).
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=${1:?outdir}; FRAMES=${2:-5}
TAG=e14b_all
[ -e "$OUT/ncu/$TAG.ncu-rep.zst" ] && { echo "REFUSING: $OUT/ncu/$TAG exists"; exit 2; }
mkdir -p "$OUT/ncu" "$OUT/raw"
if pgrep -x ncu >/dev/null || pgrep -x nsys >/dev/null; then echo "REFUSING: another profiler is running"; exit 2; fi
PY=$HOME/venvs/pi0/bin/python
R="$OUT/ncu/$TAG.sh"
cat > "$R" <<EOR
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) --profile-from-start off --nvtx --nvtx-include "regex:.*_(prefix|expert|vision|vlm)/" \
  --graph-profiling graph --cache-control none \
  --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum \
  --export "$OUT/ncu/$TAG" --force-overwrite \
  $PY bench/e14b_padtrim.py --mode ncu --frames $FRAMES
EOR
sudo -n /bin/sh "$R" > "$OUT/ncu/$TAG.stdout" 2>&1; RC=$?
sudo -n chown -R "$(id -u):$(id -g)" "$OUT" /tmp/triton_yanggon /tmp/torchinductor_yanggon "$HOME/.triton" 2>/dev/null
echo "  ncu rc=$RC"
[ -f "$OUT/ncu/$TAG.ncu-rep" ] || { echo "  NO REPORT"; exit 3; }
sudo -n "$(which ncu)" --import "$OUT/ncu/$TAG.ncu-rep" --csv --page raw > "$OUT/ncu/${TAG}_raw.csv" 2>/dev/null
zstd -19 -q -f "$OUT/ncu/$TAG.ncu-rep" -o "$OUT/ncu/$TAG.ncu-rep.zst" && rm -f "$OUT/ncu/$TAG.ncu-rep"
$PY bench/e14_bytes_analyze.py "$OUT/ncu/${TAG}_raw.csv" "$OUT/dram_bytes.csv" $FRAMES
