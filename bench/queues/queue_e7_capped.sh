#!/usr/bin/env bash
# E7 at a CAPPED power mode (MODE_30W id 2, or MODE_15W id 1), run by a human after a reboot.
#
# The agent cannot start these: both modes change TPC_PG_MASK, which is applied at boot, so
# nvpmodel asks for a reboot and AGENTS.md forbids the agent confirming one. MODE_50W needed
# no reboot because its mask is identical to MAXN's, which is why it ran unattended.
#
# What each mode actually changes, from /etc/nvpmodel.conf:
#     MAXN        mask 0    16 SMs   EMC -1            GPU 1300.5 MHz
#     MODE_50W    mask 0    16 SMs   EMC -1            GPU  816.0 MHz   <- frequency only
#     MODE_30W    mask 240   8 SMs   EMC -1
#     MODE_15W    mask 248   6 SMs   EMC 2133 MHz      <- SMs AND memory
# These are three different perturbations, not one sweep. The report must say which knob
# moved; a "power budget vs latency" table over all four conflates them.
#
# Scope, T004 step 4b: four power loops for eager + best at N=10 only, plus one E3-style
# latency run at N=4 and N=10. Nothing from a capped mode is ever averaged with MAXN.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=$(date +%Y%m%d)
PY=$HOME/venvs/pi0/bin/python
SEC=60
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
have() { [ -s "$1/power.csv" ]; }
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

# Which capped mode are we actually in? Refuse anything else: a run started in MAXN would
# silently produce MAXN numbers in a directory named for a capped mode.
PM=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
case "$PM" in
  *MODE_30W*) TAG=30W ;;
  *MODE_15W*) TAG=15W ;;
  *MAXN*) echo "REFUSING: device is in MAXN. This queue is only for MODE_30W or MODE_15W."
          echo "  Set the mode, confirm the reboot prompt YOURSELF, reboot, re-apply"
          echo "  jetson_clocks, then run this again."; exit 2 ;;
  "") echo "REFUSING: could not read the power mode at all. Check 'sudo nvpmodel -q'."
      exit 2 ;;
  *)  # Readable, but not a mode this queue handles -- say which it is rather than claiming
      # it was unreadable. MODE_50W lands here and that is correct: it needs no reboot and
      # bench/queues/queue_e7_50w.sh already covers it.
      echo "REFUSING: power mode is '$PM'. This queue is for MODE_30W or MODE_15W only."
      case "$PM" in
        *MODE_50W*) echo "  MODE_50W needs no reboot and is handled by"
                    echo "  bench/queues/queue_e7_50w.sh -- use that instead." ;;
      esac
      exit 2 ;;
esac
echo "power mode confirmed: $PM  (tag $TAG)"

# Clocks must be re-applied after every reboot, and the gate must show them locked.
sudo -n jetson_clocks < /dev/null 2>/dev/null
sudo -n jetson_clocks --fan < /dev/null 2>/dev/null
$PY - <<'PYG'
import sys; sys.path.insert(0, "bench")
import pi0_common as P, json
g = P.gate(require_maxn=False)
print("  gate:", json.dumps(g))
if not g.get("jetson_clocks_locked"):
    raise SystemExit("REFUSING: clocks are not locked (min != max). Run sudo jetson_clocks.")
print(f"  sm_count={g['sm_count']}  tpc_pg_mask={g['tpc_pg_mask']}  "
      f"gpu={int(g['gpu_cur_freq_hz'])/1e6:.1f} MHz")
PYG
[ $? -eq 0 ] || exit 3

echo "E7_${TAG}_START $(date -Is)"

for pair in "pi0 bf16_both" "pi05 bf16_both_graphs" "smolvla vlm_graphs"; do
  set -- $pair; m="$1"; rung="$2"
  out="results/${DATE}_E7_${m}_N10_${TAG}"
  have "${out}_eager_MODE_${TAG}" || run "E7 ${TAG} $m eager N=10" \
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps 10 --seconds "$SEC" --modes eager --allow-capped
  have "${out}_${rung}_MODE_${TAG}" || run "E7 ${TAG} $m best N=10" \
    $PY bench/power_phases.py --model "$m" --outdir-prefix "$out" \
        --steps 10 --seconds "$SEC" --rung "$rung" --allow-capped
done

groot() {
  local ver="$1" venv="$2" root="$3" rung="$4"; shift 4
  local out="results/${DATE}_E7_groot_${ver}_${rung}_${TAG}"
  have "$out" && { echo "########## groot_$ver $rung $TAG already done"; return; }
  echo "########## $(date +%H:%M) E7 ${TAG} groot_$ver $rung"
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

# E3-style latency at N=4 and N=10 (T004 step 4b), pi0 only -- the E3 harness is pi0's.
for n in 4 10; do
  out="results/${DATE}_E3_pi0_eager_N${n}_${TAG}"
  [ -s "$out/latency.csv" ] || run "E3 ${TAG} pi0 eager N=$n" \
    $PY bench/run_e3.py --outdir "$out" --steps "$n" --allow-capped
done

echo "E7_${TAG}_DONE $(date -Is)"
echo
echo "If BOTH 30W and 15W are now done, return to MAXN:  sudo nvpmodel -m 0 && sudo reboot"
echo "Then: sudo jetson_clocks && sudo jetson_clocks --fan, and check STATUS.md."
