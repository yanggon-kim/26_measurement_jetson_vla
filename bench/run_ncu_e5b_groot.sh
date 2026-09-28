#!/usr/bin/env bash
# E5b for GR00T (T009 step 3): the instruction-level capture of run_ncu_e5b.sh, with the
# GR00T target of run_ncu_c1_groot.sh.
#
# Two files already existed and each had half of what this needs:
#   run_ncu_e5b.sh       the E5b metric set -- per-PIPE issue, LSU request counts, all 14
#                        stall ratios, launch/occupancy limits, InstructionStats -- which ncu
#                        does NOT report unless every one is named (the trap that cost C1 two
#                        captures). But its target is ncu_target.py, LeRobot only.
#   run_ncu_c1_groot.sh  GR00T's checkout, venv, embodiment tag, API and `best` mode flags.
#                        But only C1's four sections.
# Copying the metric list and the GR00T config verbatim keeps both comparable with what
# they were copied from: the targets were picked from C1's GR00T data, so the capture must
# be of the configuration C1 measured, and the metrics must be E5b's so the analysis chain
# (crosscheck_sass.py, analyze_e5b.py) reads them the same way.
#
# Region is ae_total: GR00T's denoising loop has no per-step wrap point (see
# run_ncu_c1_groot.sh), so the NVTX range covers the whole loop and -c caps the count.
#
# usage: run_ncu_e5b_groot.sh <n17|n16|n15> <eager|best> <region> <outdir> [kernel-regex]
set -u
VER="${1:?version}"; MODE="${2:?mode}"; REGION="${3:?region}"; OUT="${4:?outdir}"
KREGEX="${5:-}"
CAP="${NCU_COUNT:-20}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
mkdir -p "$OUT/ncu" "$OUT/raw/cubin"

case "$VER" in
  n17) CKPT=nvidia/GR00T-N1.7-3B; ROOT=/home/yanggon/00_VLA/Isaac-GR00T
       VENV=$HOME/venvs/groot;   TAGARG="--embodiment-tag oxe_droid_relative_eef_relative_joint"
       APIARG="" ; TRC=$HOME/.cache/triton_groot ;;
  n16) CKPT=nvidia/GR00T-N1.6-3B; ROOT=/home/yanggon/00_VLA/Isaac-GR00T-n1d6
       VENV=$HOME/venvs/groot16; TAGARG="--embodiment-tag robocasa_panda_omron"
       APIARG="" ; TRC=$HOME/.cache/triton_groot16 ;;
  n15) CKPT=nvidia/GR00T-N1.5-3B; ROOT=/home/yanggon/00_VLA/Isaac-GR00T-n1d5
       VENV=$HOME/venvs/groot16; TAGARG="--embodiment-tag oxe_droid --data-config oxe_droid"
       APIARG="--api old" ; TRC=$HOME/.cache/triton_groot16 ;;
  *) echo "version must be n17|n16|n15"; exit 2 ;;
esac
case "$MODE" in
  eager) MODEARGS="--tf32 on" ;;
  best)  MODEARGS="--tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs" ;;
  *) echo "mode must be eager|best"; exit 2 ;;
esac

# Kernel in the TAG: every target of one (version, mode) shares the region, so without it
# each capture would overwrite the last (run_ncu_e5b.sh learned this the hard way).
KSAFE=$(printf '%s' "${KREGEX:-all}" | tr -c 'A-Za-z0-9_' '_' | cut -c1-60)
TAG="groot_${VER}_${MODE}_${REGION}_${KSAFE}"
REP="$OUT/ncu/${TAG}"
RUNNER="$OUT/ncu/${TAG}.sh"

if $HOME/venvs/pi0/bin/python - <<'GUARDPY'
import sys, os
from pathlib import Path
me = os.getpid()
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    try:
        argv = [a for a in (d / 'cmdline').read_bytes().split(b'\0') if a]
    except OSError:
        continue
    if not argv:
        continue
    if b'--import' in argv:          # `ncu --import` only reads a report file (CPU)
        continue
    if argv[0].rsplit(b'/', 1)[-1] in (b'ncu', b'nsys') or any(
            a.endswith(b'ncu_target.py') or a.endswith(b'groot_phases.py')
            for a in argv[1:]):
        print(f"  another profiler is live: {d.name} "
              f"{b' '.join(argv[:3]).decode('utf-8', 'replace')[:90]}")
        sys.exit(0)
sys.exit(1)
GUARDPY
then
  echo "  REFUSING to start $TAG — another profiler is live (sudo; it runs as root)"
  exit 3
fi

PIPES=""
for p in lsu fma alu xu adu cbu fp16; do
  PIPES="${PIPES}sm__inst_executed_pipe_${p}.avg.pct_of_peak_sustained_active,"
done
LSU="l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
STALLS=""
for s in long_scoreboard short_scoreboard wait no_instruction math_pipe_throttle \
         mio_throttle lg_throttle barrier membar drain dispatch_stall imc_miss misc \
         selected; do
  STALLS="${STALLS}smsp__average_warps_issue_stalled_${s}_per_issue_active.ratio,"
done
LAUNCH="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps"

# groot_phases.py writes its own phase files into --outdir; keep them out of the E5b
# directory's top level so they cannot be mistaken for E5b deliverables.
GPOUT="$OUT/raw/groot_phases_${TAG}"
mkdir -p "$GPOUT"

echo "########## $(date +%H:%M) ncu E5b $TAG (cap $CAP)"
t0=$SECONDS
cat > "$RUNNER" <<EOR
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=$TRC
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
. $VENV/activate_orin_venv.sh >/dev/null 2>&1
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) \\
  --profile-from-start off --nvtx --nvtx-include "${REGION}/" \\
  ${KREGEX:+--kernel-name regex:"$KREGEX"} \\
  -c $CAP \\
  --section SpeedOfLight --section WarpStateStats --section Occupancy \\
  --section InstructionStats \\
  --metrics ${PIPES}${LSU},${STALLS}${LAUNCH} \\
  --export "$REP" --force-overwrite \\
  $VENV/bin/python bench/groot_phases.py \\
    --outdir $GPOUT --checkpoint $CKPT --version $VER --gr00t-root $ROOT \\
    $APIARG $TAGARG --warmup 8 --timed 1 --rung ncu_$REGION \\
    --ncu-region $REGION $MODEARGS
EOR
sudo -n /bin/sh "$RUNNER" > "$OUT/ncu/${TAG}.stdout" 2>&1
RC_NCU=$?
# CACHE_OWNERSHIP_FIX (2026-09-25): the profiled process runs as root, so every Triton / Inductor
# cache entry it creates is root-owned; a later user-run process that hits one fails with
# PermissionError (all six compiled GR00T N-sweep runs failed that way). Hand them back.
sudo -n chown -R "$(id -u):$(id -g)" /tmp/triton_yanggon /tmp/torchinductor_yanggon \
    "$HOME/.triton" "$HOME/.cache/triton_groot" "$HOME/.cache/triton_groot16" 2>/dev/null
( exit $RC_NCU )
echo "  ncu rc=$? in $((SECONDS-t0))s"
sudo -n chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null

if [ -f "${REP}.ncu-rep" ]; then
  sudo -n "$(which ncu)" --import "${REP}.ncu-rep" --csv --page details \
       > "$OUT/ncu/${TAG}_details.csv" 2>/dev/null
  sudo -n chown "$(id -u):$(id -g)" "${REP}.ncu-rep" "$OUT/ncu/${TAG}_details.csv" 2>/dev/null
  RAW=$(stat -c%s "${REP}.ncu-rep")
  zstd -19 -q -f "${REP}.ncu-rep" -o "${REP}.ncu-rep.zst" 2>/dev/null
  ZST=$(stat -c%s "${REP}.ncu-rep.zst" 2>/dev/null || echo 0)
  echo "  rep $(numfmt --to=iec $RAW) -> zst $(numfmt --to=iec $ZST)"
  [ "$ZST" -gt 52428800 ] && echo "  OVER 50 MB COMPRESSED — lower NCU_COUNT before committing"
  rm -f "${REP}.ncu-rep"
  echo "  csv rows=$(tail -n +2 "$OUT/ncu/${TAG}_details.csv" 2>/dev/null | wc -l)"
  if grep -q "sm__inst_executed_pipe_lsu" "$OUT/ncu/${TAG}_details.csv" 2>/dev/null; then
    echo "  pipe metrics present"
  else
    echo "  *** PIPE METRICS ABSENT — this capture cannot answer E5b's question ***"
  fi
else
  echo "  NO REPORT WRITTEN — see $OUT/ncu/${TAG}.stdout"
  echo "  *** ncu matched NO kernel for regex '${KREGEX:-?}'. Targets come from C1 captures;"
  echo "      autotuning may pick different kernels on a later run. Re-derive the target. ***"
  tail -5 "$OUT/ncu/${TAG}.stdout" 2>/dev/null | sed 's/^/    /'
  exit 3
fi

n=0
for f in $(find "$TRC" -name "*.cubin" -newermt "@$(( $(date +%s) - (SECONDS - t0) - 60 ))" 2>/dev/null); do
  cp -n "$f" "$OUT/raw/cubin/" 2>/dev/null && n=$((n+1))
done
echo "  collected $n cubin file(s) into $OUT/raw/cubin/"
