#!/usr/bin/env bash
# T005 — drive ncu over bench/ncu_validation.py's four cases and collect the sysmem-miss
# counters plus kernel duration. ncu needs root on this device (T001).
#
#   bash bench/run_ncu_validation.sh results/<date>_E2_ae_kernels
set -uo pipefail
OUT="${1:?usage: run_ncu_validation.sh <result-dir>}"
PY="${PY:-$HOME/venvs/pi0/bin/python}"
NCU="$(command -v ncu)"
mkdir -p "$OUT/ncu" "$OUT/raw"
echo "case,exit,gpu_cur_freq_hz,gpu_min_freq_hz,gpu_max_freq_hz,emc_rate_hz,gpu_thermal_millideg,cache_control" > "$OUT/raw/ncu_clock_check.csv"

METRICS=$(tr -d ' \n' <<'M'
lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum,
lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum,
lts__t_sectors_aperture_sysmem_lookup_miss.sum,
lts__t_sectors_aperture_sysmem_op_read.sum,
lts__t_sectors_aperture_sysmem_op_write.sum,
lts__t_sectors_aperture_sysmem_lookup_hit.sum,
gpu__time_duration.sum,
gpc__cycles_elapsed.avg.per_second
M
)

for CASE in copy_256mib gemm_cold gemm_warm gemm_l2 gemm_l2_twice; do
  # Cold cases want ncu's default flush; warm cases must keep the caches, or ncu's own
  # invalidation removes the effect being measured.
  case "$CASE" in
    gemm_l2_twice)     CACHE="--cache-control none"; LAUNCHES=2 ;;
    gemm_warm|gemm_l2) CACHE="--cache-control none"; LAUNCHES=1 ;;
    *)                 CACHE=""; LAUNCHES=1 ;;
  esac
  echo "########## $CASE  (cache-control: ${CACHE:-default/flush})"
  # analytic bytes come from the target itself, on stdout
  "$PY" bench/ncu_validation.py --case "$CASE" > "$OUT/raw/ncu_${CASE}_analytic.json" 2>&1

  sudo -n "$NCU" \
      --profile-from-start off --nvtx --nvtx-include "micro/" \
      -c $LAUNCHES $CACHE \
      --metrics "$METRICS" \
      --csv --page raw \
      --log-file "$OUT/ncu/${CASE}.csv" \
      --export "$OUT/ncu/${CASE}.ncu-rep" --force-overwrite \
      "$PY" bench/ncu_validation.py --case "$CASE" \
      > "$OUT/raw/ncu_${CASE}_stdout.txt" 2>&1
  RC=$?
  echo "  exit=$RC rows=$(grep -c , "$OUT/ncu/${CASE}.csv" 2>/dev/null || echo 0)"
  # protocol §3.2: the per-run clock check must be COMMITTED, not just echoed.
  FREQ=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq)
  MINF=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/min_freq)
  MAXF=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq)
  EMC=$(sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate' 2>/dev/null)
  GPUC=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null)
  echo "$CASE,$RC,$FREQ,$MINF,$MAXF,$EMC,$GPUC,${CACHE:-default}" >> "$OUT/raw/ncu_clock_check.csv"
  echo "  gpu_cur_freq=$FREQ emc=$EMC"
done
echo "done -> $OUT/ncu"
