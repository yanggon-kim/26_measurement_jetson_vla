#!/usr/bin/env bash
# C3's last piece: MEASURE S for the GR00T family (the action head's token count), rather
# than taking action_horizon from config. pi0's chunk_size is 50 and its measured S is 51,
# because a state token is prepended, so the config value is a prior and not the answer.
# Chained after the C1 repair pass so it never shares the GPU with a capture.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
echo "C3S_START $(date -Is)"
while ! grep -q C1_REPAIR_DONE scratch/T004/queue_c1_repair.log 2>/dev/null; do sleep 60; done
sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "TreeLauncherSubreaper"; pkill -f "bench/ncu_target.py"; sleep 2; true' 2>/dev/null

run_s() {  # ver venv root extra...
  local ver="$1" venv="$2" root="$3"; shift 3
  echo "########## $(date +%H:%M) measure S, GR00T $ver"
  bash -c "source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --outdir results/20260923_E4f_groot_S_$ver --version $ver \
             --gr00t-root $root --warmup 4 --timed 1 --rung measure_s --measure-s $*"
  echo "EXIT=$?"
}
run_s n17 groot   /home/yanggon/00_VLA/Isaac-GR00T \
      --checkpoint nvidia/GR00T-N1.7-3B --embodiment-tag oxe_droid_relative_eef_relative_joint
run_s n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
run_s n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-config oxe_droid
echo "C3S_DONE $(date -Is)"
