#!/usr/bin/env bash
# E7 energy at MODE_50W. Unattended, and it is the last MAXN-class item before the two
# capped modes that need a human reboot.
#
# ## Why this mode can run without a reboot, checked rather than assumed
#
# /etc/nvpmodel.conf on this board:
#     POWER_MODEL ID=0 NAME=MAXN      TPC_PG_MASK 0
#     POWER_MODEL ID=1 NAME=MODE_15W  TPC_PG_MASK 248
#     POWER_MODEL ID=2 NAME=MODE_30W  TPC_PG_MASK 240
#     POWER_MODEL ID=3 NAME=MODE_50W  TPC_PG_MASK 0
#
# TPC power gating is applied at boot, which is why changing it needs a reboot. MODE_50W has
# the SAME mask as MAXN, so `sudo nvpmodel -m 3` switched with no prompt and exit 0, and the
# return to MAXN should be equally free. 30W and 15W gate TPCs (240 -> 8 SMs, 248 -> 6 SMs)
# and are a human handoff.
#
# ## What this mode actually changes, measured after the switch
#
#     MAXN       16 SMs, mask 0, GPU 1300.5 MHz, CPU 2201.6 MHz
#     MODE_50W   16 SMs, mask 0, GPU  816.0 MHz, CPU 1497.6 MHz
#
# All 16 SMs stay online. So MAXN vs 50W is a PURE FREQUENCY comparison at constant compute
# width -- better controlled than 30W/15W will be, where SM count changes too. Say so in the
# report: a 50W energy figure is not "the same machine with a lower budget", it is the same
# 16 SMs at 63 % of the clock.
#
# ## Scope, per T004 step 4b for a capped session
#
# eager + best at N=10 only, and nothing from a capped mode is ever averaged with MAXN --
# the directory name carries the mode.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
SEC=60
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
have() { [ -s "$1/power_phases.csv" ] || [ -s "$1/power.csv" ]; }
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

# Refuse if the device is not actually in 50W: every number here would carry the wrong mode.
m=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
case "$m" in
  *MODE_50W*) echo "power mode confirmed: $m" ;;
  *) echo "REFUSING: power mode is '$m', expected MODE_50W"; exit 2 ;;
esac

echo "E7_50W_START $(date -Is)"

# pi0 / pi0.5 / SmolVLA, eager then best rung, N=10
for pair in "pi0 bf16_both" "pi05 bf16_both_graphs" "smolvla vlm_graphs"; do
  set -- $pair; m="$1"; rung="$2"
  out="results/${DATE}_E7_${m}_N10_50W"
  have "${out}_eager_MODE_50W" || run "E7 50W $m eager N=10" \
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps 10 --seconds "$SEC" --modes eager --allow-capped
  have "${out}_${rung}_MODE_50W" || run "E7 50W $m best N=10" \
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps 10 --seconds "$SEC" --rung "$rung" --allow-capped
done

# GR00T x3 through their own venvs
groot() {
  local ver="$1" venv="$2" root="$3" rung="$4"; shift 4
  local out="results/${DATE}_E7_groot_${ver}_${rung}_50W"
  if have "$out"; then echo "########## $(date +%H:%M) groot_$ver $rung 50W already done"; return; fi
  echo "########## $(date +%H:%M) E7 50W groot_$ver $rung"
  sweep
  bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR; \
           source \$HOME/venvs/$venv/activate_orin_venv.sh >/dev/null 2>&1; \
           cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
           exec \$HOME/venvs/$venv/bin/python bench/groot_phases.py \
             --power --seconds $SEC --outdir $out --version $ver --rung $rung \
             --allow-capped ${root:+--gr00t-root $root} $*"
  echo "EXIT=$? groot_$ver $rung"
}

groot n17 groot "" eager --checkpoint nvidia/GR00T-N1.7-3B
groot n17 groot "" best  --checkpoint nvidia/GR00T-N1.7-3B
groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 eager \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
groot n16 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 best \
      --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron
groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 eager \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-config oxe_droid
groot n15 groot16 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 best \
      --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-config oxe_droid

echo "E7_50W_DONE $(date -Is)"
echo "NOTE: the device is STILL IN MODE_50W. Returning to MAXN is a separate, deliberate"
echo "      step so a crash here cannot leave the mode ambiguous -- see STATUS.md."
