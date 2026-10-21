#!/usr/bin/env bash
# GR00T at MODE_50W, re-run under the SAME rung names the MAXN campaign used.
#
# queue_e7_50w.sh passed `eager` and `best`. The MAXN E7 campaign used:
#     n17   eager_fp32   vlm_compile
#     n16   eager_fp32   all_compile_autotune
#     n15   eager_fp32   all_compile_autotune
# and `eager` is not `eager_fp32` -- matmul_precision high vs highest, TF32 on vs off. So
# every GR00T point from that queue pairs against a MAXN baseline at a different math
# precision, and TF32's speed advantage inflates the apparent clock-insensitive fraction.
#
# It produced a FALSE falsification of the prediction registered in
# scratch/NOTE_50w_frequency_scaling.txt: n17 eager read 31.9 % clock-insensitive, above
# every LeRobot model, which looked like the relationship breaking. The comparison was never
# valid.
#
# The earlier directories are NOT deleted or renamed. They are real measurements of
# `eager`/`best`, their names carry the rung, and renaming would assert a configuration the
# run did not have.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
SEC=60
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
have() { [ -s "$1/power.csv" ]; }

pgrep -u "$USER" -f "[q]ueue_e7_50w.sh" >/dev/null 2>&1 && {
  echo "REFUSING: queue_e7_50w.sh is still running; one GPU job at a time"; exit 2; }

PM=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
case "$PM" in *MODE_50W*) echo "power mode confirmed: $PM" ;;
  *) echo "REFUSING: power mode is '$PM', expected MODE_50W"; exit 2 ;; esac

echo "E7_50WG_START $(date -Is)"

groot() {
  local ver="$1" venv="$2" root="$3" rung="$4"; shift 4
  local out="results/${DATE}_E7_groot_${ver}_${rung}_50W"
  have "$out" && { echo "########## groot_$ver $rung already done"; return; }
  echo "########## $(date +%H:%M) E7 50W groot_$ver $rung  (MAXN-matched rung)"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --power --seconds $SEC --outdir $out --version $ver --rung $rung \
             --allow-capped ${root:+--gr00t-root $root} $*"
  echo "EXIT=$? groot_$ver $rung"
  # Assert the rung actually recorded matches what was asked. The whole defect was a rung
  # name that silently meant a different precision, so verify rather than trust the flag.
  $HOME/venvs/pi0/bin/python - "$out" "$rung" <<'PYE'
import json, sys, os
out, want = sys.argv[1], sys.argv[2]
p = os.path.join(out, "meta.json")
if not os.path.exists(p):
    print(f"  VERIFY {want}: no meta.json"); raise SystemExit
m = json.load(open(p))
got, mm = m.get("rung"), m.get("matmul_precision")
# Compare against the MAXN run for the SAME rung. Checking the rung name alone is what let
# the first correction through: the name matched and matmul_precision did not, which is the
# whole confound. Verify the quantity that differs, not a proxy for it.
import glob as _g
ref = None
for d in sorted(_g.glob(f"results/*_E7_groot_*_{want}")):
    if d.endswith("_50W"):
        continue
    try:
        ref = json.load(open(os.path.join(d, "meta.json")))
        break
    except OSError:
        pass
rmm = ref.get("matmul_precision") if ref else None
ok = got == want and (rmm is None or mm == rmm)
print(f"  VERIFY {os.path.basename(out)}: rung={got!r} matmul={mm} "
      f"(MAXN baseline matmul={rmm}) -> "
      f"{'OK' if ok else '*** MISMATCH: this does not pair with the MAXN run ***'}")
PYE
}

# Copied VERBATIM from bench/queues/queue_e7_maxn.sh:118-136, not reconstructed. The rung
# NAME does not set the math precision -- `--tf32 off|on` does, and the compiled rungs also
# carry --head-mode / --vision-mode / --llm-mode / --force-gemm-autotune. My first attempt
# passed only the rung name and produced eager_fp32 with matmul_precision=high, i.e. TF32
# still ON, against a MAXN baseline at matmul_precision=highest. The confound survived the
# fix for it.
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

echo "E7_50WG_DONE $(date -Is)"
