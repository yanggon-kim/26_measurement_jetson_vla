#!/usr/bin/env bash
# The three gaps, plus the configurations that maximise the ACTION-EXPERT SHARE.
#
# Why AE share and not just e2e (owner, 2026-09-23): the PIM premise is about the action
# expert. The configuration where vision and prefix are optimised HARDEST is the one where
# the expert's share of the frame peaks, and that is the strongest case a PIM design could
# be argued from. So every run here records the share, and the report ranks by it as well
# as by latency.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot16"; mkdir -p "$TRITON_CACHE_DIR"

echo "GAPS_START $(date -Is)"
while ! grep -q COMBINED_DONE scratch/T004/queue_groot_combined.log 2>/dev/null; do sleep 45; done
echo "gap A (groot vision+llm) clear $(date -Is)"
sudo -n /bin/sh -c 'pkill -f "bench/groot_phases.py"; pkill -f "bench/ncu_target.py"; sleep 2; true' 2>/dev/null

# ---- GAP B: GEMM autotuning on GR00T's prefix ---------------------------------------
# Its compiled prefix GEMMs are genuinely memory bound (85-94 %), a different regime from
# pi0's mis-tiled mio_throttle kernel. Orin's 48 KB/block shared memory may leave no valid
# Triton config, as it did for pi0's bf16 shapes -- that is a result either way.
gb() {
  local ver="$1" venv="$2" root="$3"; shift 3
  echo "########## $(date +%H:%M) GAP B $ver all_compile_autotune"
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_FORCE_BIG_GPU=1; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --outdir results/20260923_E4d_groot_${ver}_ladder --version $ver \
             --gr00t-root $root --warmup 20 --timed 50 --rung all_compile_autotune \
             --force-gemm-autotune --tf32 on --head-mode reduce-overhead \
             --vision-mode max-autotune-no-cudagraphs \
             --llm-mode max-autotune-no-cudagraphs $*"
  echo "EXIT=$?"
}
gb n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 \
   --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
gb n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 \
   --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-config oxe_droid

# ---- GAP C: pi0.5, bf16 + graphs on the prefix ---------------------------------------
# Graphs on pi0.5's prefix were a net loss WITHOUT bf16 (871 vs 863). Untested on top of
# bf16, where the prefix is 2.16x smaller and launch cost is a larger fraction of it.
echo "########## $(date +%H:%M) GAP C pi0.5 bf16_both + prefix graphs"
$PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_both_graphs
echo "EXIT=$?"

echo "GAPS_DONE $(date -Is)"
