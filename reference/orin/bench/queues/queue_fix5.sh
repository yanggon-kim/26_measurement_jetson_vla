#!/usr/bin/env bash
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"; mkdir -p "$TRITON_CACHE_DIR"
echo "FIX5_START $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

echo "########## $(date +%H:%M) groot vlm_graphs (mark_step_begin)"
bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
         source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
           --outdir results/20260923_E4d_groot_n17_ladder \
           --checkpoint nvidia/GR00T-N1.7-3B --version n17 --warmup 20 --timed 50 \
           --rung vlm_graphs --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead"
echo "EXIT=$?"

# one more probe: with time_mlp_in/out cast, which module is fp32 NOW?
run $PY bench/pi05_dtype_probe.py --model pi05 --prep base,norms
echo "FIX5_DONE $(date -Is)"
