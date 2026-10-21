#!/usr/bin/env bash
# The combination the GR00T ladder never tested: vision AND LLM compiled together, on top
# of the graph-captured head. Each existing rung left the other half eager.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot16"; mkdir -p "$TRITON_CACHE_DIR"
echo "COMBINED_START $(date -Is)"
run() {  # ver venv root extra...
  local ver="$1" venv="$2" root="$3"; shift 3
  echo "########## $(date +%H:%M) $ver all_compile"
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --outdir results/20260923_E4d_groot_${ver}_ladder --version $ver \
             --gr00t-root $root --warmup 20 --timed 50 --rung all_compile \
             --tf32 on --head-mode reduce-overhead \
             --vision-mode max-autotune-no-cudagraphs \
             --llm-mode max-autotune-no-cudagraphs $*"
  echo "EXIT=$?"
}
run n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 \
    --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
run n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
    --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-config oxe_droid
echo "COMBINED_DONE $(date -Is)"
