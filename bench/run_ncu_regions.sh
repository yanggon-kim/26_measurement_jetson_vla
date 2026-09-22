#!/usr/bin/env bash
# T004 step 2 — one ncu run per region, smallest first (protocol §3.2 escalation order).
#
#   ae_layer[0]/  ->  ae_step[0]/  ->  vlm_prefix/  ->  vision/
#
# Hard rules from §3.2 enforced here: never unfiltered; one NVTX range per run; --set full
# (Occupancy reads 0 % under `roofline`); both pages exported in full; `sudo $(which ncu)`,
# never `sudo -E`, with HOME/HF_HOME/cache dirs passed explicitly; the report chown'd back.
# A run that exceeds ~60 minutes is stopped and covered by the §3.2 fallback instead.
#
#   bash bench/run_ncu_regions.sh <outdir> <mode> <steps> [regions...]
set -uo pipefail
OUT="${1:?usage: run_ncu_regions.sh <outdir> <mode> <steps> [regions...]}"; shift
MODE="${1:?mode}"; shift
STEPS="${1:?steps}"; shift
REGIONS=("$@")
[ ${#REGIONS[@]} -eq 0 ] && REGIONS=("ae_layer[0]" "ae_step[0]" "vlm_prefix" "vision")
PY="${PY:-$HOME/venvs/pi0/bin/python}"
NCU="$(command -v ncu)"
mkdir -p "$OUT/ncu" "$OUT/raw"
TIMEOUT_S="${TIMEOUT_S:-3600}"
EXTRA=""
[ "$MODE" != "eager" ] && EXTRA="--no-regions"     # regions.py graph-breaks a compiled model

for R in "${REGIONS[@]}"; do
  TAG=$(echo "$R" | tr -d '[]' | tr -c 'A-Za-z0-9_' '_' | sed 's/_*$//')
  echo "########## $R  (mode=$MODE, tag=$TAG)"
  FREQ_BEFORE=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq)
  START=$(date +%s)
  timeout "$TIMEOUT_S" sudo -n "$NCU" \
      --profile-from-start off --nvtx --nvtx-include "$R/" \
      -c 1 --set full \
      --csv --page details --log-file "$OUT/ncu/${TAG}_details.csv" \
      --export "$OUT/ncu/${TAG}" --force-overwrite \
      env HOME="$HOME" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
          TORCHINDUCTOR_CACHE_DIR="$PWD/scratch/T004/inductor" \
          TRITON_CACHE_DIR="$PWD/scratch/T004/inductor/triton" \
      "$PY" bench/ncu_target.py --region "$R" --mode "$MODE" --num-steps "$STEPS" $EXTRA \
      > "$OUT/raw/ncu_${TAG}_stdout.log" 2>&1
  RC=$?; ELAPSED=$(( $(date +%s) - START ))
  FREQ_AFTER=$(cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq)
  echo "  exit=$RC  ${ELAPSED}s  devfreq ${FREQ_BEFORE} -> ${FREQ_AFTER}"
  [ $RC -eq 124 ] && echo "  STOPPED at the ${TIMEOUT_S}s limit — cover this region with the §3.2 fallback"
  # the raw page holds the stall reasons and the sysmem-miss sectors (§3.3)
  if [ $RC -eq 0 ]; then
    sudo -n "$NCU" --import "$OUT/ncu/${TAG}.ncu-rep" --csv --page raw \
        > "$OUT/ncu/${TAG}_raw.csv" 2> "$OUT/raw/ncu_${TAG}_rawpage.log" || \
        echo "  raw page export failed; see raw/ncu_${TAG}_rawpage.log"
  fi
  sudo -n chown -R "$(id -un):$(id -gn)" "$OUT" 2>/dev/null
  echo "$TAG,$RC,$ELAPSED,$FREQ_BEFORE,$FREQ_AFTER,$MODE,$R" >> "$OUT/raw/ncu_clock_check.csv"
done
echo "NCU_REGIONS_DONE"
