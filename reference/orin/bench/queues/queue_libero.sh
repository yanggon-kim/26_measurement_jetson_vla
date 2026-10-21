#!/usr/bin/env bash
# Accuracy validation for Part B's numerics-changing rungs. The fp32 baseline runs FIRST
# and is the harness's own control: a wrong observation mapping and a broken policy both
# look like a low success rate, so a bf16 number means nothing unless fp32 works.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
echo "LIBERO_START $(date -Is)"
run() {  # model prep outdir
  echo "########## $(date +%H:%M) libero $1 $2"
  local t0=$SECONDS
  MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $HOME/venvs/libero/bin/python bench/libero_eval.py \
      --model "$1" --prep "$2" --suite libero_spatial --tasks 10 --episodes 3 \
      --outdir "$3"
  echo "EXIT=$? $((SECONDS-t0))s"
}
run pi0  none      results/20260923_E8q_pi0_libero_fp32
run pi0  bf16_both results/20260923_E8q_pi0_libero_bf16both
run pi05 none      results/20260923_E8q_pi05_libero_fp32
run pi05 bf16_both results/20260923_E8q_pi05_libero_bf16both
echo "LIBERO_DONE $(date -Is)"
