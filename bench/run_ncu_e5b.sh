#!/usr/bin/env bash
# E5b / T007 — instruction-level capture: the metrics C1 deliberately did not request.
#
# C1 asked "what is this phase bound by" and four sections plus the fourteen stall ratios
# answered it. E5b asks the next question -- "which instructions are issuing, and which
# pipe is the issue going to" -- and that needs the per-PIPE metrics.
#
# **ncu reports a section's metrics but NOT the per-pipe ones unless they are requested by
# name.** This is the same trap that cost C1 two captures before it was caught: the stall
# breakdown was absent from the first two because `--section WarpStateStats` does not
# carry it. So every metric below is listed explicitly, and bench/crosscheck_sass.py shows
# a zero rather than omitting a row when one is missing.
#
# The gate from docs/03_instruction_level_analysis.md §1 is applied by the CALLER, not
# here: do not run this on a kernel whose Compute (SM) Throughput is below 5 % AND whose
# duration is comparable to the launch gap -- it is launch-bound and the instruction mix
# explains nothing. C1's per-kernel data already identifies the targets, which is the real
# saving: no exploratory captures are needed.
#
# usage: run_ncu_e5b.sh <model> <mode> <region> <outdir> [kernel-regex]
set -u
MODEL="${1:?model}"; MODE="${2:?mode}"; REGION="${3:?region}"; OUT="${4:?outdir}"
KREGEX="${5:-}"
CAP="${NCU_COUNT:-20}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
mkdir -p "$OUT/ncu" "$OUT/raw/cubin"

PY=$HOME/venvs/pi0/bin/python
# `best` here means what it means in C1: --compile-mode reduce-overhead. That is NOT
# Part B's best RUNG (pi0 bf16_both, pi0.5 bf16_both_graphs, SmolVLA vlm_graphs), and the
# two must not be conflated in the write-up. It is deliberate: E5b's target kernels were
# chosen from C1's per-kernel data, so the capture has to be of the same configuration
# those kernels were measured in, or the instruction analysis would be of kernels that
# are not the ones ranked.
case "$MODE" in
  eager) TMODE=eager ;;
  best)  TMODE=reduce-overhead ;;
  *) echo "mode must be eager|best"; exit 2 ;;
esac
case "$MODEL" in
  pi0|pi05|smolvla) N=10 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac

# The TAG must include the KERNEL. Every target of one (model, mode) shares the same
# region -- ae_step_0 -- so a tag of ${MODEL}_${MODE}_${REGION} is identical for all of
# them and each capture would silently overwrite the previous one, leaving one report
# where the queue thinks there are three. Sanitised because the name goes in a filename.
KSAFE=$(printf '%s' "${KREGEX:-all}" | tr -c 'A-Za-z0-9_' '_' | cut -c1-60)
TAG="${MODEL}_${MODE}_${REGION}_${KSAFE}"
REP="$OUT/ncu/${TAG}"

# Refuse to start while another profiler is live. Killing a queue does NOT kill the ncu it
# launched -- that tree is root-owned via sudo -- so an orphan can share the GPU with the
# next capture and corrupt both. Exact argv element only; substring-matching the joined
# cmdline self-matches, which has cost this project three incidents.
if $PY - <<'GUARDPY'
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
    if argv[0].rsplit(b'/', 1)[-1] in (b'ncu', b'nsys') or any(
            a.endswith(b'ncu_target.py') or a.endswith(b'groot_phases.py')
            for a in argv[1:]):
        print(f"  another profiler is live: {d.name} "
              f"{b' '.join(argv[:3]).decode('utf-8', 'replace')[:90]}")
        sys.exit(0)
sys.exit(1)
GUARDPY
then
  echo "  REFUSING to start $TAG — kill it first (sudo; it runs as root)"
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

echo "########## $(date +%H:%M) ncu E5b $TAG (cap $CAP)"
t0=$SECONDS
# Same environment reasoning as run_ncu_c1.sh: sudo resets HOME, so the profiled process
# would look for the HF cache under root's home and try to download a gated repo, and
# TORCHINDUCTOR_CACHE_DIR would default to /tmp/torchinductor_root and recompile every
# kernel inside ncu. `sudo -n env ...` is refused because the NOPASSWD rule names
# binaries and env is not one; /bin/sh IS, so the environment is set inside a root shell.
# No token variables: protocol §3.5, because ncu reports capture the environment.
#
# TRITON/TORCHINDUCTOR caches are pointed INTO this run so the generated .cubin for every
# Triton kernel can be collected afterwards -- a Triton kernel does not exist on disk
# until it is compiled, so there is nothing to disassemble without this.
RUNNER="$OUT/ncu/${TAG}.sh"
cat > "$RUNNER" <<EOF
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRITON_CACHE_DIR=/tmp/triton_yanggon
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) \
  --profile-from-start off --nvtx --nvtx-include "${REGION}/" \
  ${KREGEX:+--kernel-name regex:"$KREGEX"} \
  -c $CAP \
  --section SpeedOfLight --section WarpStateStats --section Occupancy \
  --section InstructionStats \
  --metrics ${PIPES}${LSU},${STALLS}${LAUNCH} \
  --export "$REP" --force-overwrite \
  $PY bench/ncu_target.py --model $MODEL --mode $TMODE --warmup ${NCU_WARMUP:-8} \
      --num-steps $N --region "$REGION"
EOF
sudo -n /bin/sh "$RUNNER" > "$OUT/ncu/${TAG}.stdout" 2>&1
RC_NCU=$?
# CACHE_OWNERSHIP_FIX (2026-09-25): the profiled process runs as root, so every Triton / Inductor
# cache entry it creates is root-owned; a later user-run process that hits one fails with
# PermissionError (all six compiled GR00T N-sweep runs failed that way). Hand them back.
sudo -n chown -R "$(id -u):$(id -g)" /tmp/triton_yanggon /tmp/torchinductor_yanggon \
    "$HOME/.triton" "$HOME/.cache/triton_groot" "$HOME/.cache/triton_groot16" 2>/dev/null
( exit $RC_NCU )
RC=$?
echo "  ncu rc=$RC in $((SECONDS-t0))s"

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
  # Did the pipe metrics actually arrive? This is the check whose absence cost C1 two
  # captures: a CSV that exists and has rows can still be missing the whole point.
  if grep -q "sm__inst_executed_pipe_lsu" "$OUT/ncu/${TAG}_details.csv" 2>/dev/null; then
    echo "  pipe metrics present"
  else
    echo "  *** PIPE METRICS ABSENT — this capture cannot answer E5b's question ***"
  fi
else
  echo "  NO REPORT WRITTEN — see $OUT/ncu/${TAG}.stdout"
  # A capture that profiled NOTHING must not exit 0. pi05/best did exactly that at 22:00:
  # ncu ran, matched no kernel, printed the list of kernels it DID see (all
  # triton_poi_fused_*), wrote no report, and the queue logged EXIT=0. Only the watchdog's
  # artefact check noticed, and that is a poor place to first learn a capture did nothing.
  #
  # The cause is worth stating in the failure, because it is not a typo: the targets come
  # from the C1 captures taken days earlier, and TorchInductor's autotuning does not
  # guarantee the same kernel choice on a later run. pi05's best rung has
  # max-autotune-no-cudagraphs on pg.forward, and this time inductor generated Triton
  # kernels where C1 had recorded a cuBLAS/cutlass one. So the target list can name a kernel
  # that this run never launches.
  echo "  *** ncu matched NO kernel for regex '${KREGEX:-?}'. The kernels it DID see"
  echo "      are listed in $OUT/ncu/${TAG}.stdout. Targets come from the C1 captures, and"
  echo "      TorchInductor autotuning can choose different kernels on a later run, so a"
  echo "      target may name a kernel this run never launches. Re-derive the target from"
  echo "      that list rather than assuming the capture is retryable as-is. ***"
  exit 3
fi

# Collect the .cubin of every kernel compiled during this run. A Triton kernel is
# generated at runtime and does not exist on disk until then, so without this there is
# nothing for build_inst_inventory.py to disassemble.
n=0
for f in $(find /tmp/triton_yanggon -name "*.cubin" -newermt "@$(( $(date +%s) - (SECONDS - t0) - 60 ))" 2>/dev/null); do
  cp -n "$f" "$OUT/raw/cubin/" 2>/dev/null && n=$((n+1))
done
echo "  collected $n cubin file(s) into $OUT/raw/cubin/"
