#!/usr/bin/env bash
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"; mkdir -p "$TRITON_CACHE_DIR"
echo "FIX4_START $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

# pi0.5 B3, with only time_mlp_in/out cast (action_in_proj must stay fp32: its input is
# the externally created noise tensor)
run $PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_norms,bf16_both

# GR00T vlm_graphs, with ModelOutput subclasses cloned in place instead of rebuilt as
# plain dicts
echo "########## $(date +%H:%M) groot vlm_graphs"
bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
         source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
           --outdir results/20260923_E4d_groot_n17_ladder \
           --checkpoint nvidia/GR00T-N1.7-3B --version n17 --warmup 20 --timed 50 \
           --rung vlm_graphs --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead"
echo "EXIT=$?"
echo "FIX4_DONE $(date -Is)"
