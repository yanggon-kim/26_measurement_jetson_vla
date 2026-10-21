#!/usr/bin/env bash
# Re-run BOTH vision-compile rungs now that setuptools is installed in each GR00T venv.
# Neither venv had it, and N1.6's vis_compile failed with
#   InductorError: ModuleNotFoundError: No module named 'setuptools'
# N1.7's vis_compile failed with the flash_attn SymInt error instead, which is a real
# limitation -- but it was recorded from a venv that ALSO lacked setuptools, so it is
# re-run rather than trusted.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
echo "VISC_START $(date -Is)"
while ! grep -q N1D5_DONE scratch/T004/queue_n1d5.log 2>/dev/null; do sleep 60; done
echo "n1.5 stage clear $(date -Is)"

rm -f results/20260923_E4d_groot_n17_ladder/raw/rows/vis_compile.json
rm -f results/20260923_E4d_groot_n16_ladder/raw/rows/vis_compile.json

echo "########## $(date +%H:%M) n1.7 vis_compile"
bash -c "export TRITON_CACHE_DIR=\$HOME/.cache/triton_groot; \
         source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
           --outdir results/20260923_E4d_groot_n17_ladder --checkpoint nvidia/GR00T-N1.7-3B \
           --version n17 --warmup 20 --timed 50 --rung vis_compile --tf32 on \
           --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs"
echo "EXIT=$?"

echo "########## $(date +%H:%M) n1.6 vis_compile"
bash -c "export TRITON_CACHE_DIR=\$HOME/.cache/triton_groot16; \
         source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot16/bin/python bench/groot_phases.py \
           --outdir results/20260923_E4d_groot_n16_ladder --checkpoint nvidia/GR00T-N1.6-3B \
           --version n16 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 \
           --embodiment-tag robocasa_panda_omron \
           --warmup 20 --timed 50 --rung vis_compile --tf32 on \
           --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs"
echo "EXIT=$?"
echo "VISC_DONE $(date -Is)"
