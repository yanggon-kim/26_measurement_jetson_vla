#!/usr/bin/env bash
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
OUT=results/20260923_E4d_groot_n16_ladder
ROOT=/home/yanggon/00_VLA/Isaac-GR00T-n1d6
TAG=robocasa_panda_omron
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"; mkdir -p "$TRITON_CACHE_DIR"
echo "FIX6_START $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

# pi0.5 B3: norms' own parameters only, no embedded Linear
run $PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_norms,bf16_both

g() {
  local rung="$1"; shift
  [ -f "$OUT/raw/rows/$rung.json" ] && { echo "########## $rung already done"; return; }
  echo "########## $(date +%H:%M) n1.6 $rung $*"
  local t0=$SECONDS
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
           source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
             --outdir $OUT --checkpoint nvidia/GR00T-N1.6-3B --version n16 \
             --gr00t-root $ROOT --embodiment-tag $TAG \
             --warmup 20 --timed 50 --rung $rung $*"
  echo "EXIT=$? $((SECONDS-t0))s"
}
g eager_fp32  --tf32 off
g eager_tf32  --tf32 on
g ae_compile  --tf32 on --head-mode max-autotune-no-cudagraphs
g ae_graphs   --tf32 on --head-mode reduce-overhead
g vis_compile --tf32 on --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs
g vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs
g vlm_graphs  --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead
echo "FIX6_DONE $(date -Is)"
