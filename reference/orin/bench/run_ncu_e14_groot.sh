#!/usr/bin/env bash
# T016 step 3 (A4): DRAM bytes per frame of GR00T N1.7 per savings VARIANT (ref, keep1, kvonce, keep1+kvonce),
# by the E12b metric set (L2 fills from memory = DRAM reads; L2 write sectors = upper bound on DRAM writes),
# rung vlm_compile (head reduce-overhead = CUDA graphs, LLM max-autotune-no-cudagraphs), N = 4.
#
# One inference inside the NVTX range "frame" (model.get_action, bench/groot_phases.py --ncu-region),
# --graph-profiling graph (a CUDA-graph replay is one workload). The compiled backbone is NOT a graph, so a
# frame has hundreds of kernels, and ncu dies with LaunchFailed after 480 profiled kernels per process on this
# device (memory ncu-limits-on-orin): capture in chunks of 450 workloads (--launch-skip / --launch-count),
# one process per chunk, until a chunk returns < 450 rows. Units are taken per chunk file (bench/e14_groot_bytes.py).
# Torch.compile workers must not be ncu-injected: the Inductor cache is warm from the time_* runs and the
# compiled control repeat, and TORCHINDUCTOR_COMPILE_THREADS=1 keeps any compile in-process.
#
# usage: run_ncu_e14_groot.sh <outdir> [variants=ref,keep1,kvonce,keep1+kvonce]
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=${1:?outdir}; VARIANTS=${2:-ref,keep1,kvonce,keep1+kvonce}
CH=450
VENV=$HOME/venvs/groot
mkdir -p "$OUT/ncu" "$OUT/raw"
if pgrep -x ncu >/dev/null || pgrep -x nsys >/dev/null; then echo "REFUSING: another profiler is running"; exit 2; fi
for v in ${VARIANTS//,/ }; do
  case $v in ref) F="";; keep1) F="--keep1";; kvonce) F="--kvonce";; keep1+kvonce) F="--keep1 --kvonce";;
    *) echo "unknown variant $v"; exit 2;; esac
  s=0
  while :; do
    TAG="e14g_${v}_s$s"
    [ -e "$OUT/ncu/${TAG}_raw.csv" ] && { echo "REFUSING: $OUT/ncu/${TAG}_raw.csv exists"; exit 2; }
    R="$OUT/ncu/$TAG.sh"
    cat > "$R" <<EOR
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
TORCHINDUCTOR_COMPILE_THREADS=1
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_COMPILE_THREADS
. $VENV/activate_orin_venv.sh >/dev/null 2>&1
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) --profile-from-start off --nvtx --nvtx-include "frame/" \\
  --graph-profiling graph --cache-control none --launch-skip $s --launch-count $CH \\
  --metrics lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration.sum \\
  --export "$OUT/ncu/$TAG" --force-overwrite \\
  $VENV/bin/python bench/groot_phases.py --outdir $OUT/raw/run_$v --checkpoint nvidia/GR00T-N1.7-3B --version n17 \\
    --steps 4 --warmup 20 --timed 1 --tf32 on --rung vlm_compile --head-mode reduce-overhead \\
    --llm-mode max-autotune-no-cudagraphs $F --ncu-region frame
EOR
    echo "  $(date +%H:%M) ncu $TAG"
    t0=$SECONDS
    timeout 2400 sudo -n /bin/sh "$R" > "$OUT/ncu/$TAG.stdout" 2>&1; RC=$?
    sudo -n chown -R "$(id -u):$(id -g)" "$OUT" /tmp/triton_yanggon /tmp/torchinductor_yanggon "$HOME/.triton" 2>/dev/null
    echo "    rc=$RC in $((SECONDS-t0))s"
    [ -f "$OUT/ncu/$TAG.ncu-rep" ] || { echo "  NO REPORT for $TAG"; tail -5 "$OUT/ncu/$TAG.stdout"; exit 3; }
    sudo -n "$(which ncu)" --import "$OUT/ncu/$TAG.ncu-rep" --csv --page raw > "$OUT/ncu/${TAG}_raw.csv" 2>/dev/null
    sudo -n chown "$(id -u):$(id -g)" "$OUT/ncu/${TAG}_raw.csv" "$OUT/ncu/$TAG.ncu-rep" 2>/dev/null
    zstd -19 -q -f "$OUT/ncu/$TAG.ncu-rep" -o "$OUT/ncu/$TAG.ncu-rep.zst" && rm -f "$OUT/ncu/$TAG.ncu-rep"
    n=$(( $(wc -l < "$OUT/ncu/${TAG}_raw.csv") - 2 ))
    echo "    workloads $n"
    [ "$n" -gt 0 ] || { echo "  EMPTY chunk $TAG (NVTX filter matched nothing?)"; exit 3; }
    [ "$n" -lt $CH ] && break
    s=$((s + CH))
  done
done
$HOME/venvs/pi0/bin/python bench/e14_groot_bytes.py "$OUT"
