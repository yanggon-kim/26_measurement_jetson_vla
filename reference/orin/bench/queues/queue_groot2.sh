#!/usr/bin/env bash
# GR00T N1.7, re-run under the THREE-WAY split (vision / llm prefix / action head), plus
# the vlm_compile_nvg rung pi0 and pi0.5 are missing. Waits for pi0.5's B3 fix.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
OUT=results/20260923_E4d_groot_n17_ladder

export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"      # ~/.triton is root-owned
mkdir -p "$TRITON_CACHE_DIR"

echo "GROOT2_START $(date -Is)"
while ! grep -q PI05_B3FIX_DONE scratch/T004/queue_pi05_b3_fix.log 2>/dev/null; do sleep 45; done
echo "pi05 clear $(date -Is)"

run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

# The four existing GR00T rows were measured with vlm_prefix = the WHOLE backbone. The
# instrumentation now times backbone.model.visual as its own region and derives the prefix
# as backbone minus vision, so those rows measure a different quantity under the same name
# and are removed rather than mixed with the new ones.
rm -f $OUT/raw/rows/*.json

g() {
  local rung="$1"; shift
  if [ -f "$OUT/raw/rows/$rung.json" ]; then echo "########## $rung already done"; return; fi
  echo "########## $(date +%H:%M) groot $rung $*"
  local t0=$SECONDS
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
           source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
             --outdir $OUT --checkpoint nvidia/GR00T-N1.7-3B --version n17 \
             --warmup 20 --timed 50 --rung $rung $*"
  echo "EXIT=$? $((SECONDS-t0))s"
}

g eager_fp32  --tf32 off
g eager_tf32  --tf32 on
g ae_compile  --tf32 on --head-mode max-autotune-no-cudagraphs
g ae_graphs   --tf32 on --head-mode reduce-overhead
# expected to fail: Qwen3-VL's vision encoder passes int(grid_thw.max().item()) to
# flash_attn_varlen_func, which dynamo cannot trace. It now leaves a row saying so.
g vis_compile --tf32 on --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs
g vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs
g vlm_graphs  --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead

# the rung pi0 and pi0.5 never got, so the cross-model table has the same columns
for m in pi0 pi05; do
  run $PY bench/opt_ladder.py --model $m --outdir results/20260922_E4c_${m}_opt_ladder \
      --steps 10 --warmup 20 --timed 10 --rungs vlm_compile_nvg
done

echo "GROOT2_DONE $(date -Is)"
