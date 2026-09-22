#!/usr/bin/env bash
# T010 / T004 — capture a short NVTX-annotated nsys trace and export the CSVs we analyse.
#
# Protocol §3.1: a capture window of 1-3 steady-state inferences, not a whole run. The target
# script warms up and then wraps the measured inferences in cudaProfilerStart/Stop, so the
# capture range is exactly those inferences.
#
# T001 gotcha, recorded in results/20260921_E0_platform/meta.json: plain
# `--capture-range=cudaProfilerApi` KILLS the target at cudaProfilerStop, so
# `--capture-range-end=stop` is mandatory.
#
#   bash bench/run_nsys.sh <outdir> <tag> -- <python> <script> [args...]
set -uo pipefail
OUT="${1:?usage: run_nsys.sh <outdir> <tag> -- <cmd...>}"; shift
TAG="${1:?missing tag}"; shift
[ "${1:-}" = "--" ] && shift
mkdir -p "$OUT/raw" "$OUT/nsys"
REP="$OUT/raw/${TAG}"

nsys profile \
    -t cuda,nvtx,osrt,cublas \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-memory-usage=true \
    -f true -o "$REP" \
    "$@" > "$OUT/raw/${TAG}_stdout.log" 2>&1
RC=$?
echo "nsys exit=$RC -> ${REP}.nsys-rep"
[ $RC -ne 0 ] && tail -5 "$OUT/raw/${TAG}_stdout.log"

# protocol §3.2: record the clock next to every capture
echo "gpu_cur_freq_hz=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq)" \
     >> "$OUT/raw/${TAG}_stdout.log"

for R in cuda_gpu_kern_sum cuda_gpu_trace nvtx_sum nvtx_gpu_proj_trace cuda_api_sum; do
  nsys stats -r "$R" -f csv -o "$OUT/nsys/${TAG}" --force-export=true "${REP}.nsys-rep" \
      >> "$OUT/raw/${TAG}_stats.log" 2>&1 \
    && echo "  exported $R" || echo "  $R unavailable on this nsys build"
done
ls -1 "$OUT/nsys/" 2>/dev/null | sed 's/^/  /'
