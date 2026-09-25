#!/usr/bin/env bash
# T015 / E12b queue (owner-directed 2026-09-25). One GPU job at a time; MAXN only (NN5).
# Stages run in spec order; each is its own result directory (spec §11 naming).
#   bash bench/queues/queue_e12b.sh <stage>     stages: m2 nsys m3 m4calib m4rung m4green m5 mech all
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
# immutable snapshot (bash reads scripts incrementally; see memory never-edit-running-bash-script)
if [ -z "${QP_SNAPSHOT:-}" ]; then
  snap="scratch/queues/queue_e12b.${1:-none}.$(date +%s).sh"; mkdir -p scratch/queues
  cp "$0" "$snap"; QP_SNAPSHOT=1 exec bash "$snap" "$@"
fi
PY=$HOME/venvs/pi0/bin/python
D=${DATE:-$(date +%Y%m%d)}
M1=$(ls -d results/*_E12b_pi0_fairexpert_N10_MAXN 2>/dev/null | tail -1)
gate() {
  local pm; pm=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
  case "$pm" in *MAXN*) : ;; *) echo "REFUSING: power mode '$pm' is not MAXN (NN5)"; exit 2 ;; esac
  sudo -n jetson_clocks >/dev/null 2>&1 || true
  sudo -n /bin/sh -c 'tegrastats --stop; pkill -9 -x tegrastats' >/dev/null 2>&1 || true
}
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; gate; "$@"; echo "EXIT=$? $l"; }
st() {
  case "$1" in
    m2)   run "E12b M2 compiled co-run, full GPU" $PY -u bench/e12b_m2.py \
            --outdir results/${D}_E12b_pi0_corun_N10_MAXN --m1 "$M1/summary.json" ;;
    nsys) mkdir -p scratch/e12b/nsys
          run "E12b M2 nsys timeline" env -i HOME=$HOME PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/bin:/bin \
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon TRITON_CACHE_DIR=/tmp/triton_yanggon \
            /usr/local/bin/nsys profile --capture-range=cudaProfilerApi \
            --capture-range-end=stop --cuda-graph-trace=node -t cuda,nvtx --force-overwrite true \
            -o scratch/e12b/nsys/m2 $PY -u bench/e12b_nsys.py
          d=results/${D}_E12b_pi0_corun_N10_MAXN/nsys; mkdir -p "$d"
          for rep in cuda_gpu_trace nvtx_pushpop_trace cuda_api_sum cuda_gpu_kern_sum; do
            /usr/local/bin/nsys stats --report $rep --format csv --force-export true \
              -o "$d/m2" scratch/e12b/nsys/m2.nsys-rep >/dev/null 2>&1
          done
          ls "$d"
          $PY bench/e12b_nsys_analyze.py --gpu-trace "$d/m2_cuda_gpu_trace.csv" \
            --nvtx "$d/m2_nvtx_pushpop_trace.csv" --out results/${D}_E12b_pi0_corun_N10_MAXN
          echo "EXIT=$? nsys analysis" ;;
    m3)   run "E12b M3 green co-run + hog" $PY -u bench/e12b_m3.py \
            --outdir results/${D}_E12b_pi0_green_corun_N10_MAXN ;;
    m4calib) run "E12b M4 EMC calibration" $PY -u bench/e12b_m4.py --mode calib \
            --outdir results/${D}_E12b_pi0_emc_calibration_MAXN ;;
    m4rung)  run "E12b M4 bandwidth, M2 config" $PY -u bench/e12b_m4.py --mode rung \
            --outdir results/${D}_E12b_pi0_corun_N10_MAXN \
            --calib results/${D}_E12b_pi0_emc_calibration_MAXN/emc_calibration.json ;;
    m4green) run "E12b M4 bandwidth, M3 config" $PY -u bench/e12b_m4.py --mode graphs_both \
            --split "${BEST_SPLIT:-12/4}" --outdir results/${D}_E12b_pi0_green_corun_N10_MAXN \
            --calib results/${D}_E12b_pi0_emc_calibration_MAXN/emc_calibration.json ;;
    m5)   run "E12b M5 energy" $PY -u bench/e12b_m5.py --outdir results/${D}_E12b_pi0_energy_N10_MAXN \
            --corun-config "${M5_CONFIG:-rung_prefix}" ${M5_SPLIT:+--corun-split $M5_SPLIT} ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}
if [ "$1" = all ]; then
  for s in m2 nsys m3 m4calib m4rung m4green m5; do st $s; done
  echo "E12B_DONE $(date -Is)"
else
  st "$1"
fi
