#!/usr/bin/env bash
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=results/20260923_E4d_groot_n15_ladder
ROOT=/home/yanggon/00_VLA/Isaac-GR00T-n1d5
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot16"; mkdir -p "$TRITON_CACHE_DIR"
echo "N15LADDER_START $(date -Is)"
g() {
  local rung="$1"; shift
  [ -f "$OUT/raw/rows/$rung.json" ] && { echo "########## $rung already done"; return; }
  echo "########## $(date +%H:%M) n1.5 $rung $*"
  local t0=$SECONDS
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
           source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot16/bin/python bench/groot_phases.py \
             --outdir $OUT --checkpoint nvidia/GR00T-N1.5-3B --version n15 \
             --gr00t-root $ROOT --api old --embodiment-tag oxe_droid \
             --data-config oxe_droid --warmup 20 --timed 50 --rung $rung $*"
  echo "EXIT=$? $((SECONDS-t0))s"
}
g eager_fp32  --tf32 off
g eager_tf32  --tf32 on
g ae_compile  --tf32 on --head-mode max-autotune-no-cudagraphs
g ae_graphs   --tf32 on --head-mode reduce-overhead
g vis_compile --tf32 on --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs
g vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs
g vlm_graphs  --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead
echo "N15LADDER_DONE $(date -Is)"
