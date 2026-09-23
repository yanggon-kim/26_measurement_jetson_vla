#!/usr/bin/env bash
# C1 — what is each phase bound by? One ncu capture per (model, mode, region).
#
# WHY NOT `--set full`: the overnight C1 run used `--set full -c 300` and produced four
# .ncu-rep files of 89, 184, 201 and 203 MB. Protocol §6 caps a committed file at 50 MB and
# GitHub rejects anything over 100 MB, so none of them could be committed. C1's question is
# narrow -- memory bound, compute bound, or latency bound, and on which stall -- and four
# sections answer it:
#
#   SpeedOfLight           SM [%] vs Memory [%]  -> compute- vs memory-bound
#   MemoryWorkloadAnalysis DRAM/L2/L1 throughput -> which level is saturated
#   WarpStateStats         stall reasons         -> Long Scoreboard = memory latency
#   Occupancy              achieved vs theoretical
#
# Protocol §3.2 is still enforced: never unfiltered, one NVTX range per run, with -c.
#
# usage: run_ncu_c1.sh <model> <mode> <region> <outdir>
#   model   pi0 | pi05 | smolvla          (resolved through bench/model_registry.py)
#   mode    eager | best                  (best = reduce-overhead)
#   region  vision | vlm_prefix | ae_step_0
set -u
MODEL="${1:?model}"; MODE="${2:?mode}"; REGION="${3:?region}"; OUT="${4:?outdir}"
CAP="${NCU_COUNT:-60}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
mkdir -p "$OUT/ncu"

PY=$HOME/venvs/pi0/bin/python
case "$MODE" in
  eager) TMODE=eager ;;
  best)  TMODE=reduce-overhead ;;
  *) echo "mode must be eager|best"; exit 2 ;;
esac

# N per model: its own configured default, the same one Part A/B used.
case "$MODEL" in
  pi0|pi05) N=10 ;;
  smolvla)  N=10 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac

# bracket-free alias: ncu's --nvtx-include expression language gives [ and ] special
# meaning, and two 250 s runs profiled ZERO kernels before this was understood.
NVTX_R="$REGION"
TAG="${MODEL}_${MODE}_${REGION}"
REP="$OUT/ncu/${TAG}"

echo "########## $(date +%H:%M) ncu $TAG (cap $CAP)"
t0=$SECONDS
# sudo resets HOME, so the profiled process looked for the HF cache under root's home,
# found nothing, and tried to DOWNLOAD a gated repo:
#   "Cannot access gated repo ... google/paligemma-3b-pt-224 ... You must be authenticated"
# Point it at the real cache and force offline, so it reads the weights already on disk and
# never contacts the hub. Deliberately NOT passing HF_TOKEN: protocol §3.5 requires profiled
# commands to run without token variables, because ncu reports capture the environment.
# `sudo -n env ...` is refused ("a password is required"): the NOPASSWD rule names
# specific binaries, and env is not one of them. /bin/sh IS, so the environment is set
# inside a root shell instead. Deliberately NOT passing HF_TOKEN: protocol §3.5 requires
# profiled commands to run without token variables, because ncu reports capture the
# environment. Offline mode means no token is needed at all -- the weights are on disk.
RUNNER="$OUT/ncu/${TAG}.sh"
cat > "$RUNNER" <<EOF
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) \
  --profile-from-start off --nvtx --nvtx-include "${NVTX_R}/" \
  -c $CAP \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section WarpStateStats --section Occupancy \
  --export "$REP" --force-overwrite \
  $PY bench/ncu_target.py --model $MODEL --mode $TMODE --warmup ${NCU_WARMUP:-8} \
      --num-steps $N --region "$REGION"
EOF
sudo -n /bin/sh "$RUNNER" > "$OUT/ncu/${TAG}.stdout" 2>&1
RC=$?
echo "  ncu rc=$RC in $((SECONDS-t0))s"

if [ -f "${REP}.ncu-rep" ]; then
  # CSV export is what the analysis reads; the .ncu-rep is the raw artefact.
  sudo -n "$(which ncu)" --import "${REP}.ncu-rep" --csv --page details \
       > "$OUT/ncu/${TAG}_details.csv" 2>/dev/null
  sudo -n chown "$(id -u):$(id -g)" "${REP}.ncu-rep" "$OUT/ncu/${TAG}_details.csv" 2>/dev/null
  RAW=$(stat -c%s "${REP}.ncu-rep")
  zstd -19 -q -f "${REP}.ncu-rep" -o "${REP}.ncu-rep.zst" 2>/dev/null
  ZST=$(stat -c%s "${REP}.ncu-rep.zst" 2>/dev/null || echo 0)
  echo "  rep $(numfmt --to=iec $RAW) -> zst $(numfmt --to=iec $ZST)"
  # Protocol §6: keep every committed file under 50 MB. Say so loudly if we are over.
  if [ "$ZST" -gt 52428800 ]; then
    echo "  OVER 50 MB COMPRESSED — shorten the capture (lower NCU_COUNT) before committing"
  fi
  rm -f "${REP}.ncu-rep"     # only the compressed one is kept, per §6
  KERNELS=$(tail -n +2 "$OUT/ncu/${TAG}_details.csv" 2>/dev/null | wc -l)
  echo "  csv rows=$KERNELS"
else
  echo "  NO REPORT WRITTEN — see $OUT/ncu/${TAG}.stdout"
  tail -5 "$OUT/ncu/${TAG}.stdout" 2>/dev/null | sed 's/^/    /'
fi
