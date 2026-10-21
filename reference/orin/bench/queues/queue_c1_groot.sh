#!/usr/bin/env bash
# C1 for the GR00T family, chained behind the LeRobot matrix so two profilers never run.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export NCU_COUNT=40
echo "C1_GROOT_START $(date -Is)"
while ! grep -q C1_LEROBOT_DONE scratch/T004/queue_c1_lerobot.log 2>/dev/null; do sleep 60; done
echo "lerobot C1 clear $(date -Is)"

# Sweep orphans before starting: killing a queue does not kill its root-owned ncu.
sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "TreeLauncherSubreaper"; pkill -f "bench/ncu_target.py"; pkill -f "bench/groot_phases.py"; sleep 2; true' 2>/dev/null

# Smoke-test the --ncu-region path ONCE before spending 18 captures on it. It has never
# run: groot_phases.py grew NVTX ranges + cudaProfilerStart/Stop for C1 and nothing has
# exercised them. If it is broken, every GR00T capture fails and the repair pass fails the
# same way. This runs alone (the LeRobot matrix is finished by here), without ncu, so it
# costs ~2 minutes and never shares the GPU with a capture.
echo "########## $(date +%H:%M) smoke test: --ncu-region on N1.7"
if ! bash -c 'source $HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1
      cd /home/yanggon/00_VLA/26_measurement_jetson_vla
      exec $HOME/venvs/groot/bin/python bench/groot_phases.py         --outdir /tmp/c1_groot_smoke --checkpoint nvidia/GR00T-N1.7-3B --version n17         --embodiment-tag oxe_droid_relative_eef_relative_joint         --warmup 4 --timed 1 --rung smoke --ncu-region vision --tf32 on'       > scratch/T004/c1_groot_smoke.log 2>&1; then
  echo "SMOKE TEST FAILED — not starting 18 captures. Tail:"
  tail -12 scratch/T004/c1_groot_smoke.log | sed 's/^/    /'
  echo "C1_GROOT_DONE $(date -Is)  (aborted after smoke test)"
  exit 1
fi
echo "  smoke test passed"

for VER in n17 n16 n15; do
  for MODE in eager best; do
    for REGION in vision vlm_prefix ae_total; do
      OUT=results/20260923_E4g_c1_groot_${VER}_${MODE}
      # "done" means USABLE, not merely present. An orphaned ncu from a killed
      # queue once finished after its directory was moved aside and wrote a CSV
      # with 0 stall metrics and 130 rows; a [ -f ] check called that done.
      if $HOME/venvs/pi0/bin/python bench/c1_capture_ok.py \
           "$OUT/ncu/groot_${VER}_${MODE}_${REGION}_details.csv" >/dev/null 2>&1; then
        echo "########## $VER $MODE $REGION already done (verified)"; continue
      fi
      # Sweep before EVERY capture, not just at queue start. An ncu orphaned by
      # a previous kill is root-owned and outlives its queue, and the runner
      # guard then correctly refuses to start -- but a refusal used to end the
      # whole queue, which is how smolvla/best/ae_step_0 was skipped.
      sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "TreeLauncherSubreaper"; pkill -f "bench/ncu_target.py"; pkill -f "bench/groot_phases.py"; sleep 2; true' 2>/dev/null
      bash bench/run_ncu_c1_groot.sh "$VER" "$MODE" "$REGION" "$OUT"
    done
  done
done
echo "C1_GROOT_DONE $(date -Is)"
