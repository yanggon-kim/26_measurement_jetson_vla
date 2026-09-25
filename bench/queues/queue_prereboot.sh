#!/usr/bin/env bash
# Pre-reboot completion queue (owner, 2026-09-25): every remaining item that needs no reboot.
# Plan: /home/yanggon/.claude/plans/you-are-the-main-temporal-island.md
#
# Run one stage at a time:  bash bench/queues/queue_prereboot.sh <stage>
# Each stage logs to scratch/queues/queue_prereboot_<stage>.log with the ########/EXIT=/_DONE
# markers bench/watchdog.sh §3 parses. One GPU job at a time; never two profilers at once.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
# Run from an IMMUTABLE SNAPSHOT. bash reads a script incrementally while executing it, so
# editing this file while a stage runs shifts the offsets under the running shell: on
# 2026-09-25 09:21 that produced "line 85: the: command not found" and dropped the e5b_groot
# stage straight into the e6_rungs stage, which had been appended to the file minutes earlier.
if [ -z "${QP_SNAPSHOT:-}" ]; then
  snap="scratch/queues/queue_prereboot.${1:-none}.$(date +%s).sh"
  mkdir -p scratch/queues; cp "$0" "$snap"
  QP_SNAPSHOT=1 exec bash "$snap" "$@"
fi
PY=$HOME/venvs/pi0/bin/python
export DATE=${DATE:-$(date +%Y%m%d)}
STAGE="${1:?stage}"

gate() {  # refuse unless MAXN -- a run started capped would carry MAXN-looking names
  local pm; pm=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
  case "$pm" in *MAXN*) : ;; *) echo "REFUSING: power mode is '$pm', not MAXN"; exit 2 ;; esac
  sudo -n jetson_clocks >/dev/null 2>&1 || true
}
sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; pkill -f "[n]sys profile"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

case "$STAGE" in
# ---------------------------------------------------------------------------------------
e5b_groot)
  gate
  echo "E5B_GROOT_START $(date -Is)"
  TMP=scratch/queues/e5b_groot_targets.sh
  $PY bench/pick_e5b_targets.py --only-groot --emit-queue > "$TMP"
  echo "  $(wc -l < "$TMP") capture(s) planned"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    # skip a target whose report already exists (the first was captured by hand at 09:14
    # to validate the script); never overwrite a capture
    k=$(printf '%s' "$line" | sed -E 's/.*"([^"]*)"$/\1/' | tr -c 'A-Za-z0-9_' '_' | cut -c1-60)
    d=$(printf '%s' "$line" | awk '{print $7}' | sed "s/\${DATE}/$DATE/")
    if ls "$d"/ncu/*_"$k"*.ncu-rep.zst >/dev/null 2>&1; then
      echo "  already captured: $d $k"; continue
    fi
    run "E5b groot capture: $k" eval "$line"
  done < "$TMP"

  # SASS straight from the reports (what actually executed; the cuBLAS kernels live in
  # libcublasLt and have no .cubin on disk)
  for rep in results/${DATE}_E5b_groot_*/ncu/*.ncu-rep.zst; do
    [ -e "$rep" ] || continue
    d=$(dirname "$(dirname "$rep")")
    echo "########## $(date +%H:%M) SASS $(basename "$rep")"
    $PY bench/ncu_sass_dump.py "$rep" --outdir "$d/raw/sass"; echo "EXIT=$? SASS $(basename "$rep")"
  done
  for sass in results/${DATE}_E5b_groot_*/raw/sass/*.sass; do
    [ -e "$sass" ] || continue
    d=$(dirname "$(dirname "$(dirname "$sass")")"); k=$(basename "$sass" .sass)
    mkdir -p "$d/inventory"
    $PY bench/build_inst_inventory.py "$sass" --label "$k" --csv > "$d/inventory/${k}_inventory.csv" \
      && $PY bench/build_inst_inventory.py "$sass" --label "$k" > "$d/inventory/${k}_inventory.md"
    echo "EXIT=$? inventory $k"
  done
  # Cross-check each inventory against the details CSV of the SAME kernel. queue_e5b.sh took
  # `ls ncu/*_details.csv | head -1`, which in a directory holding several targets pairs a
  # kernel with another kernel's metrics.
  for inv in results/${DATE}_E5b_groot_*/inventory/*_inventory.csv; do
    [ -e "$inv" ] || continue
    d=$(dirname "$(dirname "$inv")"); k=$(basename "$inv" _inventory.csv)
    ks=$(printf '%s' "$k" | tr -c 'A-Za-z0-9_' '_' | cut -c1-40)
    det=$(ls "$d"/ncu/*"${ks}"*_details.csv 2>/dev/null | head -1)
    if [ -z "$det" ]; then
      echo "  NO details CSV matching kernel $k in $d -- cross-check skipped, not guessed"
      continue
    fi
    $PY bench/crosscheck_sass.py --inventory "$inv" --ncu-csv "$det" --target-kernel "$k" \
        --gpu-arch sm_87 > "$d/inventory/${k}_crosscheck.md"
    echo "EXIT=$? crosscheck $k"
  done
  # Its OWN report. analyze_e5b.py's default --md is reports/E5b_summary.md, which holds the
  # six LeRobot configurations and must not be overwritten.
  $PY bench/analyze_e5b.py --glob "results/${DATE}_E5b_groot_*" --md reports/E5b_groot_analysis.md
  echo "EXIT=$? analyze_e5b groot"
  echo "E5B_GROOT_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e5b_groot_sweep)
  # T009 step 3's S sweep, docs/03 "S sweep (isolated kernels)": S in {1, 16, 41 (N1.7's own
  # S, per T009), 256} on the DiT's largest projection and its q projection.
  gate
  echo "E5B_SWEEP_START $(date -Is)"
  OUT=results/${DATE}_E5b_groot_n17_ssweep; mkdir -p "$OUT/ncu" "$OUT/raw"
  GV=$HOME/venvs/groot
  run "sweep shapes (groot venv, reads the loaded model)" /bin/bash -c \
    ". $GV/activate_orin_venv.sh >/dev/null 2>&1; HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     $GV/bin/python bench/micro_proj_sweep.py --emit-shapes $OUT/shapes.json \
       --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T"
  [ -s "$OUT/shapes.json" ] || { echo "REFUSING: no shapes.json -- shapes are read, never typed"; exit 3; }
  run "sweep timing (CUDA events)" $PY bench/micro_proj_sweep.py --shapes "$OUT/shapes.json" \
      --S 1,16,41,256 --reps 200 --outdir "$OUT"
  PIPES=""; for q in lsu fma alu xu adu cbu fp16; do PIPES="${PIPES}sm__inst_executed_pipe_${q}.avg.pct_of_peak_sustained_active,"; done
  STALLS=""; for q in long_scoreboard short_scoreboard wait no_instruction math_pipe_throttle mio_throttle lg_throttle barrier membar drain dispatch_stall imc_miss misc selected; do STALLS="${STALLS}smsp__average_warps_issue_stalled_${q}_per_issue_active.ratio,"; done
  for key in largest q; do for S in 1 16 41 256; do
    TAG="sweep_${key}_S${S}"; R="$OUT/ncu/${TAG}.sh"
    cat > "$R" <<EOR
HOME=/home/yanggon
export HOME
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
exec $(which ncu) --profile-from-start off --nvtx --nvtx-include "${key}_S${S}/" -c 3 \
  --section SpeedOfLight --section WarpStateStats --section Occupancy --section InstructionStats \
  --metrics ${PIPES}${STALLS}launch__grid_size,launch__block_size,launch__registers_per_thread \
  --export "$OUT/ncu/${TAG}" --force-overwrite \
  $PY bench/micro_proj_sweep.py --shapes $OUT/shapes.json --ncu-window ${key}:${S}
EOR
    echo "########## $(date +%H:%M) ncu $TAG"; sweep
    sudo -n /bin/sh "$R" > "$OUT/ncu/${TAG}.stdout" 2>&1; rc=$?
    sudo -n chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null
    if [ -f "$OUT/ncu/${TAG}.ncu-rep" ]; then
      sudo -n "$(which ncu)" --import "$OUT/ncu/${TAG}.ncu-rep" --csv --page details > "$OUT/ncu/${TAG}_details.csv" 2>/dev/null
      zstd -19 -q -f "$OUT/ncu/${TAG}.ncu-rep" -o "$OUT/ncu/${TAG}.ncu-rep.zst" && rm -f "$OUT/ncu/${TAG}.ncu-rep"
    else rc=3; echo "  NO REPORT for $TAG"; fi
    echo "EXIT=$rc ncu $TAG"
  done; done
  echo "E5B_SWEEP_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e6_rungs)
  # Matched-rung E6 (plan step 4, and the fix for the mixed-rung comparison corrected in
  # 6d27541). Every model at its eager rung AND its best rung, region times from Part B's
  # UNPROFILED ladder rows, bytes counted from the model AS PREPARED FOR THAT RUNG.
  gate
  echo "E6_RUNGS_START $(date -Is)"
  OUT=results/${DATE}_E6_rungs; mkdir -p "$OUT"
  for spec in "pi0 eager_tf32 20260922_E4c_pi0_opt_ladder" "pi0 bf16_both 20260922_E4c_pi0_opt_ladder" \
              "pi05 eager_tf32 20260922_E4c_pi05_opt_ladder" "pi05 bf16_both_graphs 20260922_E4c_pi05_opt_ladder" \
              "smolvla eager_tf32 20260922_E4c_smolvla_opt_ladder" "smolvla vlm_graphs 20260922_E4c_smolvla_opt_ladder"; do
    set -- $spec; m=$1; r=$2; lad=$3
    d="$OUT/${m}_${r}"
    if [ -s "$d/roofline.csv" ]; then echo "  done already: $d"; continue; fi
    if [ "$r" = "eager_tf32" ]; then rarg=""; else rarg="--rung $r"; fi
    run "floors $m $r" $PY bench/floors_models.py --model $m --steps 10 $rarg --outdir "$d"
    vf=""; [ -f scratch/vision_flops_$m.json ] && vf="--vision-flops scratch/vision_flops_$m.json"
    run "roofline $m $r" $PY bench/roofline.py --outdir "$d" --floor "$d/floor.csv" \
        --ladder-row results/$lad/raw/rows/${r}_N10.json $vf --model $m
    [ -n "$vf" ] && cp "scratch/vision_flops_$m.json" "$d/vision_flops.json"
  done
  # GR00T: bytes do not change across its rungs (they set tf32 and compilation, not weights),
  # so N1.7's floor is reused for every rung and only the ladder row differs.
  for r in eager_tf32 ae_compile vlm_compile; do
    d="$OUT/groot_n17_${r}"; mkdir -p "$d"
    [ -s "$d/roofline.csv" ] && { echo "  done already: $d"; continue; }
    cp results/20260925_E6_groot_n17/floor.csv results/20260925_E6_groot_n17/floor_inputs.json "$d/"
    run "roofline groot_n17 $r" $PY bench/roofline.py --outdir "$d" --floor "$d/floor.csv" \
        --ladder-row results/20260923_E4d_groot_n17_ladder/raw/rows/${r}.json --model groot_n17
  done
  echo "E6_RUNGS_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
