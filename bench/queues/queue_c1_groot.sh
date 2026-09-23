#!/usr/bin/env bash
# C1 for the GR00T family, chained behind the LeRobot matrix so two profilers never run.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export NCU_COUNT=40
echo "C1_GROOT_START $(date -Is)"
while ! grep -q C1_LEROBOT_DONE scratch/T004/queue_c1_lerobot.log 2>/dev/null; do sleep 60; done
echo "lerobot C1 clear $(date -Is)"

# Sweep orphans before starting: killing a queue does not kill its root-owned ncu.
sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "bench/ncu_target.py"; pkill -f "bench/groot_phases.py"; sleep 2; true' 2>/dev/null

for VER in n17 n16 n15; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_total; do
      OUT=results/20260923_E4g_c1_groot_${VER}_${MODE}
      if [ -f "$OUT/ncu/groot_${VER}_${MODE}_${REGION}.ncu-rep.zst" ]; then
        echo "########## $VER $MODE $REGION already done"; continue
      fi
      bash bench/run_ncu_c1_groot.sh "$VER" "$MODE" "$REGION" "$OUT"
    done
  done
done
echo "C1_GROOT_DONE $(date -Is)"
