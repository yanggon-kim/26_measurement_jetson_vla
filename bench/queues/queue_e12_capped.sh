#!/usr/bin/env bash
# E12 Step 4 at a CAPPED power mode, run by a human after a reboot (T014 step 4 / spec §5).
#
# The agent cannot start this: MODE_30W and MODE_15W change TPC_PG_MASK, which is applied at
# boot, so nvpmodel asks for a reboot and AGENTS.md forbids the agent confirming one.
#
#     MAXN       mask 0     16 SMs   EMC -1        GPU 1300.5 MHz
#     MODE_30W   mask 240    8 SMs   EMC -1
#     MODE_15W   mask 248    6 SMs   EMC 2133 MHz
#
# ---------------------------------------------------------------------------------------
# A PRE-REGISTERED PREDICTION, written BEFORE the run, from the MAXN green-context data in
# results/20260925_E12_pi0_green_N10_MAXN_eager/sm_sensitivity.csv.
#
# On 2026-09-25 CUDA green contexts were used to give pi0 a SOFTWARE partition of the SM
# array at MAXN. MODE_30W gates 8 SMs in HARDWARE at boot. These should agree, and if they
# do it validates the green-context method against real silicon; if they disagree it means a
# software SM partition is not equivalent to a gated one, which is worth knowing on its own.
#
# MAXN, green context, pi0 eager N=10, median ms:
#       8 SMs   prefix 779.79   expert 587.56
#      16 SMs   prefix 478.08   expert 580.79
#   so at 8 SMs: prefix 1.631x, expert 1.012x
#
# PREDICTION for MODE_30W (8 SMs in hardware), eager N=10:
#   * IF MODE_30W's GPU clock equals MAXN's 1300.5 MHz, the whole-GPU times at MODE_30W
#     should match the 8-SM green-context numbers above to within a few percent.
#   * IF its clock is lower, both should be SLOWER than those numbers by roughly the clock
#     ratio -- so record `cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq` FIRST, which
#     step 0 below does, or the comparison cannot be made at all.
#   * The EXPERT should remain nearly insensitive: it lost only 1.2 % going 16 -> 8 SMs, so
#     at MODE_30W its slowdown versus MAXN should be explained almost entirely by clock, not
#     by SM count. If the expert slows far more than the clock ratio, the green-context
#     result is wrong and E12 §5's conclusion needs revisiting.
#   * issue/wall should stay ~1.0000. Fewer SMs does not make the host issue faster.
# ---------------------------------------------------------------------------------------
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=$(date +%Y%m%d)
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

PM=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
case "$PM" in
  *MODE_30W*) TAG=MODE_30W; SMS=8 ;;
  *MODE_15W*) TAG=MODE_15W; SMS=6 ;;
  *MAXN*) echo "REFUSING: device is in MAXN. This queue is only for MODE_30W / MODE_15W."
          echo "  The MAXN results already exist:"
          echo "    results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync"
          echo "    results/20260925_E12_pi0_green_N10_MAXN_{eager,bf16both}"; exit 2 ;;
  "") echo "REFUSING: cannot read the power mode. Check 'sudo nvpmodel -q'."; exit 2 ;;
  *)  echo "REFUSING: power mode is '$PM', not one this queue handles."; exit 2 ;;
esac
echo "E12_CAPPED_START $(date -Is)  mode=$TAG"

# Step 0: the device state the prediction above needs. Without the clock the MAXN comparison
# is not interpretable, so this is recorded before anything is measured.
mkdir -p scratch/queues
{ echo "mode: $PM"
  echo "expected_sm_count: $SMS"
  echo "gpu_cur_freq_Hz: $(cat /sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq 2>/dev/null)"
  echo "gpu_max_freq_Hz: $(cat /sys/devices/gpu.0/devfreq/17000000.gpu/max_freq 2>/dev/null)"
  # E12b spec §14: the eager expert is host-bound and MODE_30W slows the CPU too, so the CPU
  # clocks must be recorded for the MAXN comparison to be interpretable (MAXN reference: 12
  # cores at 2201600 kHz, 2026-09-25).
  for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    echo "cpu_cur_freq_kHz $(basename $(dirname $(dirname $c))): $(cat $c)"; done
  echo "maxn_reference_gpu_Hz: 1300500000"
  echo "date: $(date -Is)"
} | tee "scratch/queues/e12_${TAG}_device_state.txt"

sudo -n jetson_clocks 2>/dev/null || echo "  NOTE: jetson_clocks failed; record that in STATUS"

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

# 0b. The GPU-BOUND cross-check first (E12b spec §14): the GEMM control, whose MAXN 8-SM green
#     value is 0.891 ms/GEMM. This, not the eager expert, is the fair test of whether a software
#     8-SM partition behaves like 8 hardware SMs, after scaling by the GPU clock alone.
run "GEMM control $TAG (GPU-bound cross-check)" $PY bench/green_ctx.py

# 1. Step 4 proper: the same pipeline measured at MAXN, so s / p / r / T_pipe are comparable.
#    --nosync and eager match results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync exactly;
#    changing either would make the capped numbers incomparable with the MAXN ones.
run "E12 corun N=10 $TAG eager nosync" \
  $PY bench/corun.py --mode timing --path eager --nosync --steps 10 --frames 100 --warm 20 \
     --outdir "results/${DATE}_E12_pi0_corun_N10_${TAG}_eager_nosync"

# 2. The equivalence gate again. Non-negotiable 1 is not inherited from another power mode:
#    fewer SMs changes the schedule, and nothing downstream is valid until this passes here.
run "E12 equivalence $TAG" \
  $PY bench/corun.py --mode equivalence --path eager --nosync --steps 10 --frames 6 \
     --outdir "results/${DATE}_E12_pi0_equiv_${TAG}"

# 3. The green-context SM curve INSIDE the capped mode. At MODE_30W only 8 SMs exist, so the
#    requestable partitions are smaller; the harness records what the driver actually granted
#    and never assumes it got what it asked for. This is the arm that tests whether a software
#    partition of 8 SMs behaves like a hardware gate to 8.
run "E12 green SM curve $TAG eager" \
  $PY bench/corun_green.py --rung eager --steps 10 --sms 2,4,6,8 --seconds 20 \
     --outdir "results/${DATE}_E12_pi0_green_N10_${TAG}_eager"

# 4. Energy at the capped mode, same two arms as MAXN.
run "E12 energy $TAG" \
  $PY bench/corun_energy.py --steps 10 --frames 10 --seconds 60 \
     --outdir "results/${DATE}_E12_pi0_energy_N10_${TAG}_eager_nosync"

echo "E12_CAPPED_DONE $(date -Is)"
echo
echo "After this, restore the device: sudo nvpmodel -m 0, confirm the reboot YOURSELF,"
echo "reboot, then sudo jetson_clocks. Check the §1 gate reads MAXN / 16 SMs before any"
echo "further MAXN work -- a MAXN run started at 8 SMs would be silently wrong."
