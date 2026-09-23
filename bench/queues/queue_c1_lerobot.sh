#!/usr/bin/env bash
# C1 for the three LeRobot models: eager and best, three regions each.
# Pilot: 485 s per capture, 22 MB raw -> 919 KB zstd, well inside protocol 6's 50 MB.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export NCU_COUNT=40 NCU_WARMUP=8
# Sweep orphaned profilers before starting anything. Killing a queue does NOT kill
# the ncu it launched -- that tree is root-owned via sudo, so kill returns "Operation
# not permitted" -- so an orphan can outlive its queue and silently share the GPU with
# the next capture. Owner, 2026-09-23: never two profilers at once.
sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "bench/ncu_target.py"; sleep 2; true' 2>/dev/null
echo "C1_LEROBOT_START $(date -Is)"
for MODEL in pi0 pi05 smolvla; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_step_0; do
      OUT=results/20260923_E4g_c1_${MODEL}_${MODE}
      # "done" means USABLE, not merely present. An orphaned ncu from a killed
      # queue once finished after its directory was moved aside and wrote a CSV
      # with 0 stall metrics and 130 rows; a [ -f ] check called that done.
      if $HOME/venvs/pi0/bin/python bench/c1_capture_ok.py \
           "$OUT/ncu/${MODEL}_${MODE}_${REGION}_details.csv" >/dev/null 2>&1; then
        echo "########## $MODEL $MODE $REGION already done (verified)"; continue
      fi
      bash bench/run_ncu_c1.sh "$MODEL" "$MODE" "$REGION" "$OUT"
    done
  done
done
echo "C1_LEROBOT_DONE $(date -Is)"
