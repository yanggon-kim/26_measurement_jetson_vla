#!/usr/bin/env bash
# E5 nsys timeline: six models, eager and best, with the per-region split intact.
#
# ## What already exists, and is not re-measured
#
# pi0's EAGER timeline is committed and its headline claim is settled:
# results/20260922_E5_pi0_eager_N{4,10}/timeline_summary.csv give gpu_idle_frac
# 0.8427 / 0.8417, kernels_per_unit 968.25 / 968.7, gap median 62.4 / 61.6 us. This queue
# does not repeat them.
#
# ## What was missing, and why it looked impossible
#
# The compiled half. results/20260922_E5_pi0_reduce-overhead_N4/ holds a 0-byte stdout and
# no CSVs at all. The reason it was abandoned: regions.py's context managers graph-break a
# model compiled through --compile-mode, so compiled captures ran --no-regions and had no
# NVTX ranges to project the timeline onto.
#
# That is true of --compile-mode, which is applied to `sample_actions` and swallows the
# whole frame. It is NOT true of the ladder rungs. The rungs compile embed_image,
# container.forward and denoise_step individually, and the regions are instrumented AFTER
# compiling, so every region boundary stays OUTSIDE a compiled callable. That ordering is
# exactly why Part B HAS per-region numbers for its compiled rungs. So the compiled
# timeline is measurable after all, per region, by asking for a rung instead of a mode.
#
# Each model's `best` rung is the one Part B reports, so the timeline refers to a latency
# number that is already published.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

# Bracketed first character so the pattern does not match the sh that runs it.
sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; pkill -f "[n]sys profile"; sleep 2; true' 2>/dev/null
}

# "done" means the exported CSV exists and has rows, not that a directory was created --
# the abandoned compiled run left a directory and a 0-byte log behind.
have() { [ -s "$1" ]; }

echo "E5_START $(date -Is)"

# ---- the LeRobot family ----------------------------------------------------------------
# $1 model  $2 N  $3 tag  $4.. extra args to nsys_target
lerobot() {
  local m="$1" n="$2" tag="$3"; shift 3
  local out="results/${DATE}_E5_${m}_${tag}_N${n}"
  if have "$out/nsys/${tag}_N${n}_cuda_gpu_trace.csv"; then
    echo "########## $(date +%H:%M) $m $tag N=$n already done"; return
  fi
  echo "########## $(date +%H:%M) E5 $m $tag N=$n"
  sweep
  bash bench/run_nsys.sh "$out" "${tag}_N${n}" -- \
      "$PY" bench/nsys_target.py --model "$m" --steps "$n" --inferences 2 "$@"
  echo "EXIT=$?"
}

# $1 version  $2 venv  $3 root  $4 tag  then the rung's own flags
groot() {
  local ver="$1" venv="$2" root="$3" tag="$4"; shift 4
  local out="results/${DATE}_E5_groot_${ver}_${tag}"
  if have "$out/nsys/${tag}_cuda_gpu_trace.csv"; then
    echo "########## $(date +%H:%M) groot_$ver $tag already done"; return
  fi
  echo "########## $(date +%H:%M) E5 groot_$ver $tag"
  sweep
  # groot_phases --ncu-region installs the same three NVTX ranges and profiles between
  # cudaProfilerStart/Stop, which is what nsys --capture-range=cudaProfilerApi needs. For
  # nsys the range NAME is only a label: nsys captures the whole window and uses the
  # ranges to split the timeline, where ncu filters on one of them.
  bash bench/run_nsys.sh "$out" "$tag" -- \
      /bin/bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
        source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
        cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
        exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
          --outdir $out --version $ver --ncu-region all --profile-inferences 2 \
          ${root:+--gr00t-root $root} $*"
  echo "EXIT=$?"
}

# pi0: the eager half is already committed for N=4 and N=10, so only the compiled half.
lerobot pi0      4 best  --rung bf16_both
lerobot pi0     10 best  --rung bf16_both
lerobot pi05    10 eager
lerobot pi05    10 best  --rung bf16_both_graphs
lerobot smolvla 10 eager
lerobot smolvla 10 best  --rung vlm_graphs

groot n17 groot "" eager --checkpoint nvidia/GR00T-N1.7-3B --tf32 off
groot n17 groot "" best  --checkpoint nvidia/GR00T-N1.7-3B --tf32 on \
      --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs

groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 eager \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron --tf32 off
groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 best \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron \
      --force-gemm-autotune --tf32 on --head-mode reduce-overhead \
      --vision-mode max-autotune-no-cudagraphs --llm-mode max-autotune-no-cudagraphs

groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 eager \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
      --data-config oxe_droid --tf32 off
groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 best \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
      --data-config oxe_droid --force-gemm-autotune --tf32 on \
      --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs \
      --llm-mode max-autotune-no-cudagraphs

echo "E5_DONE $(date -Is)"
