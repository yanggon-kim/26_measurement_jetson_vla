#!/usr/bin/env bash
# E7 at MAXN: power and energy per phase, six models, eager and best.
#
# This is the campaign's empty column. Parts A/B/C measured latency and bottlenecks for
# six models; no power or energy number exists for any of them, and the PIM argument is
# about joules.
#
# Per run: a 60 s idle baseline in this power mode, then the prefix / ae / full loops for
# 60 s each, four INA3221 rails at 20 Hz, tegrastats alongside. Protocol §4.
#
# ## Which configurations, and why these
#
#   eager  -- the fp32 baseline every other part of the study compares against.
#   best   -- each model's fastest configuration FROM PART B, by name, so the power number
#             refers to a latency number that is already reported:
#               pi0        bf16_both             424.46 ms   (eager 1069.04)
#               pi05       bf16_both_graphs      487.81 ms   (eager 3028.56)
#               smolvla    vlm_graphs             92.30 ms   (eager  727.46)
#               groot_n17  vlm_compile           240.49 ms   (eager  372.48)
#               groot_n16  all_compile_autotune  174.59 ms   (eager  327.22)
#               groot_n15  all_compile_autotune  168.75 ms   (eager  255.22)
#
# SmolVLA's `whole_graphs` is faster still (80.97 ms) but compiles `sample_actions` as one
# unit, so it has no phase split at all -- the prefix and ae loops would run eager under it
# and their power could not be attributed. `vlm_graphs` is the fastest configuration whose
# PHASES are separable, which is what a per-phase power measurement needs. The 11 ms
# difference is recorded here rather than silently dropped.
#
# ## Why --rung and not --modes
#
# A compile mode is applied to `sample_actions`, and the prefix and ae loops call
# `embed_prefix` / `container.forward` / `denoise_step` BELOW it -- so under --modes those
# two loops run eager whatever was requested, which is how T010's parts once summed to
# 1.57x the whole. The ladder rungs compile those three callables themselves, so --rung
# reaches all three loops. The `exec_style` column in power.csv records which happened.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
SEC=${SEC:-60}
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

# Never two things on the GPU at once (owner, 2026-09-23). Profilers orphaned by a killed
# queue are root-owned via sudo and outlive it, so they are swept before anything starts.
sweep() {
  sudo -n /bin/sh -c 'pkill -f "ncu --profile-from-start"; pkill -f "TreeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}

# A run is "done" only if power.csv exists AND carries the rows it exists for. A crashed
# run can leave a directory and a meta.json behind; [ -d ] would call that done.
done_ok() {
  $PY - "$1" <<'PYEOF'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1]) / "power.csv"
if not p.exists():
    sys.exit(1)
rows = list(csv.DictReader(p.open()))
phases = {r["phase"] for r in rows}
# 3 phases x (4 rails + module + module_without_memory_rail) = 18 rows
sys.exit(0 if phases >= {"prefix", "ae", "full"} and len(rows) >= 18 else 1)
PYEOF
}

# ---- the LeRobot family ---------------------------------------------------------------
# $1 model  $2 N  $3 rung-or-"eager"
lerobot() {
  local m="$1" n="$2" rung="$3"
  local out="results/${DATE}_E7_${m}_N${n}"
  if done_ok "${out}_${rung}_MAXN"; then
    echo "########## $(date +%H:%M) $m N=$n $rung already done (verified)"; return
  fi
  echo "########## $(date +%H:%M) E7 $m N=$n $rung"
  sweep
  if [ "$rung" = "eager" ]; then
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps "$n" --seconds "$SEC" --modes eager
  else
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps "$n" --seconds "$SEC" --rung "$rung"
  fi
  echo "EXIT=$?"
}

# ---- GR00T ----------------------------------------------------------------------------
# $1 version  $2 venv  $3 gr00t-root  $4 rung  then the rung's own flags
groot() {
  local ver="$1" venv="$2" root="$3" rung="$4"; shift 4
  local out="results/${DATE}_E7_groot_${ver}_${rung}"
  if done_ok "$out"; then
    echo "########## $(date +%H:%M) groot_$ver $rung already done (verified)"; return
  fi
  echo "########## $(date +%H:%M) E7 groot_$ver $rung"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --power --seconds $SEC --outdir $out --version $ver --rung $rung \
             ${root:+--gr00t-root $root} $*"
  echo "EXIT=$?"
}

echo "E7_MAXN_START $(date -Is)"

# pi0 gets both N, because T004 step 4 asks for both and pi0 is the model the rest of the
# study is calibrated on. Every other model is measured at its own N only.
lerobot pi0     4  eager
lerobot pi0    10  eager
lerobot pi0     4  bf16_both
lerobot pi0    10  bf16_both
lerobot pi05   10  eager
lerobot pi05   10  bf16_both_graphs
lerobot smolvla 10 eager
lerobot smolvla 10 vlm_graphs

groot n17 groot "" eager_fp32 --checkpoint nvidia/GR00T-N1.7-3B --tf32 off
groot n17 groot "" vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --tf32 on \
      --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs

groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 eager_fp32 \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron --tf32 off
groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 all_compile_autotune \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron \
      --force-gemm-autotune --tf32 on --head-mode reduce-overhead \
      --vision-mode max-autotune-no-cudagraphs --llm-mode max-autotune-no-cudagraphs

groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 eager_fp32 \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
      --data-config oxe_droid --tf32 off
groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 all_compile_autotune \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid \
      --data-config oxe_droid --force-gemm-autotune --tf32 on \
      --head-mode reduce-overhead --vision-mode max-autotune-no-cudagraphs \
      --llm-mode max-autotune-no-cudagraphs

echo "E7_MAXN_DONE $(date -Is)"
