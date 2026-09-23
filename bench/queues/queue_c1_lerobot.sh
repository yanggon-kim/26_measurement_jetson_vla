#!/usr/bin/env bash
# C1 for the three LeRobot models: eager and best, three regions each.
# Pilot: 485 s per capture, 22 MB raw -> 919 KB zstd, well inside protocol 6's 50 MB.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export NCU_COUNT=40 NCU_WARMUP=8
echo "C1_LEROBOT_START $(date -Is)"
for MODEL in pi0 pi05 smolvla; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_step_0; do
      OUT=results/20260923_E4g_c1_${MODEL}_${MODE}
      if [ -f "$OUT/ncu/${MODEL}_${MODE}_${REGION}.ncu-rep.zst" ]; then
        echo "########## $MODEL $MODE $REGION already done"; continue
      fi
      bash bench/run_ncu_c1.sh "$MODEL" "$MODE" "$REGION" "$OUT"
    done
  done
done
echo "C1_LEROBOT_DONE $(date -Is)"
