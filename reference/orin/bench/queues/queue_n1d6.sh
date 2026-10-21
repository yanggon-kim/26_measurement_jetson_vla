#!/usr/bin/env bash
# GR00T N1.6, same ladder and same three-way split as N1.7.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=results/20260923_E4d_groot_n16_ladder
ROOT=/home/yanggon/00_VLA/Isaac-GR00T-n1d6
TAG=robocasa_panda_omron
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot16"; mkdir -p "$TRITON_CACHE_DIR"

echo "N1D6_START $(date -Is)"
:
echo "fix5 clear $(date -Is)"

g() {
  local rung="$1"; shift
  if [ -f "$OUT/raw/rows/$rung.json" ]; then echo "########## $rung already done"; return; fi
  echo "########## $(date +%H:%M) n1.6 $rung $*"
  local t0=$SECONDS
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
           source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot16/bin/python bench/groot_phases.py \
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
echo "N1D6_DONE $(date -Is)"
