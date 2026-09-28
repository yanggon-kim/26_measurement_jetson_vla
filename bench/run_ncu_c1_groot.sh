#!/usr/bin/env bash
# C1 for the GR00T family — same four sections and the same size discipline as
# bench/run_ncu_c1.sh, but GR00T lives in its own checkouts and environments.
#
# usage: run_ncu_c1_groot.sh <version> <mode> <region> <outdir>
#   version  n17 | n16 | n15
#   mode     eager | best        (best = the ladder's best rung for that version)
#   region   vision | vlm_prefix | ae_total
#
# ae_total, not ae_step_0: GR00T's denoising loop lives INSIDE action_head.get_action and
# has no per-step wrap point, so the range covers the whole loop and -c caps it at the
# first N kernels. Stated here because the LeRobot side profiles one step, and the two
# are therefore not the same slice of work.
#
# The stall breakdown is requested explicitly for the same reason as in
# run_ncu_c1.sh: the four sections alone do not carry it, and 'Long
# Scoreboard' is the stall reason this study needs.
set -u
VER="${1:?version}"; MODE="${2:?mode}"; REGION="${3:?region}"; OUT="${4:?outdir}"
CAP="${NCU_COUNT:-40}"
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
mkdir -p "$OUT/ncu"

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

# "best" per version, from each ladder's own lowest end-to-end rung.
case "$MODE" in
  eager) MODEARGS="--tf32 on" ;;
  best)  MODEARGS="--tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs" ;;
  *) echo "mode must be eager|best"; exit 2 ;;
esac

TAG="groot_${VER}_${MODE}_${REGION}"
REP="$OUT/ncu/${TAG}"
RUNNER="$OUT/ncu/${TAG}.sh"

# As in run_ncu_c1.sh: `sudo -n env` is refused because the NOPASSWD rule names specific
# binaries, so the environment is set inside a root shell (/bin/sh IS in that list).
# No HF_TOKEN is passed — protocol §3.5 forbids token variables in a profiled process, and
# offline mode needs none because the weights are already on disk.
# TORCHINDUCTOR_CACHE_DIR points at the user's cache so the compiled
# artefacts from Part A/B are reused instead of rebuilt inside ncu.
cat > "$RUNNER" <<EOF
HOME=/home/yanggon
HF_HOME=/home/yanggon/.cache/huggingface
HF_HUB_OFFLINE=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon
TRANSFORMERS_OFFLINE=1
TRITON_CACHE_DIR=$TRC
export HOME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
. $VENV/activate_orin_venv.sh >/dev/null 2>&1
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) \\
  --profile-from-start off --nvtx --nvtx-include "${REGION}/" \\
  -c $CAP \\
  --section SpeedOfLight --section MemoryWorkloadAnalysis \\
  --section WarpStateStats --section Occupancy \\
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio,smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,smsp__average_warps_issue_stalled_misc_per_issue_active.ratio,smsp__average_warps_issue_stalled_selected_per_issue_active.ratio \\
  --export "$REP" --force-overwrite \\
  $VENV/bin/python bench/groot_phases.py \\
    --outdir $OUT --checkpoint $CKPT --version $VER --gr00t-root $ROOT \\
    $APIARG $TAGARG --warmup 8 --timed 1 --rung ncu_$REGION \\
    --ncu-region $REGION $MODEARGS
EOF

# REFUSE to start while another profiler is live. Killing a queue does NOT kill the ncu it
# launched: that tree runs as root via sudo, so `kill` from this user returns "Operation
# not permitted" and the capture keeps going. Twice now a new capture has started while an
# orphan was still profiling, and two profilers sharing one GPU corrupt both. Checked by
# exact argv element, never a substring of the joined cmdline (that self-match has cost
# this project three incidents).
if $HOME/venvs/pi0/bin/python - <<'GUARDPY'
import sys, os
from pathlib import Path
me = os.getpid()
for d in Path('/proc').iterdir():
    if not d.name.isdigit() or int(d.name) == me:
        continue
    try:
        argv = [a for a in (d/'cmdline').read_bytes().split(b'\0') if a]
    except OSError:
        continue
    if not argv:
        continue
    if b'--import' in argv:          # `ncu --import` only reads a report file (CPU)
        continue
    if argv[0].rsplit(b'/', 1)[-1] == b'ncu' or any(
            a.endswith(b'ncu_target.py') or a.endswith(b'groot_phases.py') for a in argv[1:]):
        print(f"  another profiler is live: {d.name} {b' '.join(argv[:3]).decode()[:90]}")
        sys.exit(0)
sys.exit(1)
GUARDPY
then
  echo "  REFUSING to start $VER/$MODE/$REGION — kill it first (sudo, it runs as root)"
  exit 3
fi

echo "########## $(date +%H:%M) ncu $TAG (cap $CAP)"
t0=$SECONDS
sudo -n /bin/sh "$RUNNER" > "$OUT/ncu/${TAG}.stdout" 2>&1
RC_NCU=$?
# CACHE_OWNERSHIP_FIX (2026-09-25): the profiled process runs as root, so every Triton / Inductor
# cache entry it creates is root-owned; a later user-run process that hits one fails with
# PermissionError (all six compiled GR00T N-sweep runs failed that way). Hand them back.
sudo -n chown -R "$(id -u):$(id -g)" /tmp/triton_yanggon /tmp/torchinductor_yanggon \
    "$HOME/.triton" "$HOME/.cache/triton_groot" "$HOME/.cache/triton_groot16" 2>/dev/null
( exit $RC_NCU )
echo "  ncu rc=$? in $((SECONDS-t0))s"

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
else
  echo "  NO REPORT WRITTEN — see $OUT/ncu/${TAG}.stdout"
  tail -5 "$OUT/ncu/${TAG}.stdout" 2>/dev/null | sed 's/^/    /'
fi
