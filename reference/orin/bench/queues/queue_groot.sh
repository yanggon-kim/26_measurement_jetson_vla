#!/usr/bin/env bash
# GR00T N1.7 Part A + Part B: the same one-factor ladder the LeRobot models got.
# Waits for PARTB_FINISH_DONE so the two never share the GPU.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
OUT=results/20260923_E4d_groot_n17_ladder
LOGDIR=scratch/T004

# /home/yanggon/.triton is owned by ROOT (created 09-21 23:53 by a sudo ncu run), so
# torch 2.10's inductor cannot write its Triton driver cache and every compiled rung dies
# with PermissionError [Errno 13]. Chowning it back needs sudo, and the session-only
# password is deliberately not kept anywhere, so point the cache at a writable directory
# instead. This changes only where build artefacts are cached, never what is measured.
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"
mkdir -p "$TRITON_CACHE_DIR"

echo "GROOT_QUEUE_START $(date -Is)"
while ! grep -q PARTB_FINISH_DONE "$LOGDIR/queue_partb_finish.log" 2>/dev/null; do sleep 60; done
echo "part B clear $(date -Is)"

g() {   # g <rung> [extra args...]
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

# Part A: the paper configuration (eager) plus the TF32 control.
g eager_fp32 --tf32 off
g eager_tf32 --tf32 on

# Part B: one factor per rung, cumulative, exactly as opt_ladder.py does it.
g ae_compile --tf32 on --head-mode max-autotune-no-cudagraphs
g ae_graphs  --tf32 on --head-mode reduce-overhead
g vlm_compile --tf32 on --head-mode reduce-overhead --backbone-mode max-autotune-no-cudagraphs
g vlm_graphs  --tf32 on --head-mode reduce-overhead --backbone-mode reduce-overhead

echo "GROOT_QUEUE_DONE $(date -Is)"
