#!/usr/bin/env bash
# Final pass: re-run any C1 capture that is missing or unusable. Runs after BOTH C1 queues,
# so a capture lost to an orphan, a crash or a mid-run kill is not silently skipped.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export NCU_COUNT=40 NCU_WARMUP=8
echo "C1_REPAIR_START $(date -Is)"
while ! grep -q C1_GROOT_DONE scratch/T004/queue_c1_groot.log 2>/dev/null; do sleep 60; done
sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "TreeLauncherSubreaper"; pkill -f "bench/ncu_target.py"; pkill -f "bench/groot_phases.py"; sleep 2; true' 2>/dev/null

for MODEL in pi0 pi05 smolvla; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_step_0; do
      OUT=results/20260923_E4g_c1_${MODEL}_${MODE}
      CSV="$OUT/ncu/${MODEL}_${MODE}_${REGION}_details.csv"
      $HOME/venvs/pi0/bin/python bench/c1_capture_ok.py "$CSV" >/dev/null 2>&1 && continue
      echo "########## REPAIR $MODEL $MODE $REGION"
      bash bench/run_ncu_c1.sh "$MODEL" "$MODE" "$REGION" "$OUT"
    done
  done
done
for VER in n17 n16 n15; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_total; do
      OUT=results/20260923_E4g_c1_groot_${VER}_${MODE}
      CSV="$OUT/ncu/groot_${VER}_${MODE}_${REGION}_details.csv"
      $HOME/venvs/pi0/bin/python bench/c1_capture_ok.py "$CSV" >/dev/null 2>&1 && continue
      echo "########## REPAIR $VER $MODE $REGION"
      bash bench/run_ncu_c1_groot.sh "$VER" "$MODE" "$REGION" "$OUT"
    done
  done
done
echo "C1_REPAIR_DONE $(date -Is)"
