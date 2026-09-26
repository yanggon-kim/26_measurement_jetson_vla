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
e3e4_sweep)
  # T006 step 1 / T009 step 1: the N sweep N in {1,2,4,6,8,10} for the eager rung and the best
  # rung (per-step time = slope over N, protocol §2). The committed ladders ran N=10 only
  # (pi0.5, SmolVLA) and N=4 only (GR00T), so the slope did not exist for any of them.
  gate
  echo "E3E4_SWEEP_START $(date -Is)"
  for spec in "pi05 bf16_both_graphs" "smolvla vlm_graphs"; do
    set -- $spec; m=$1; best=$2
    d=results/${DATE}_E3E4_${m}_Nsweep
    [ -s "$d/opt_ladder.csv" ] && { echo "  done already: $d"; continue; }
    run "N sweep $m eager_tf32,$best" $PY bench/opt_ladder.py --model $m \
        --steps 1,2,4,6,8,10 --rungs eager_tf32,$best --outdir "$d"
  done
  GV=$HOME/venvs/groot
  for N in 1 2 4 6 8 10; do
    for rung in eager_tf32 vlm_compile; do
      d=results/${DATE}_E3E4_groot_n17_N${N}_${rung}
      [ -s "$d/phase_split.csv" ] && { echo "  done already: $d"; continue; }
      case $rung in
        eager_tf32)  RA="--tf32 on" ;;
        vlm_compile) RA="--tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs" ;;
      esac
      run "groot_n17 N=$N $rung" /bin/bash -c ". $GV/activate_orin_venv.sh >/dev/null 2>&1; \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_yanggon \
        TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon exec $GV/bin/python bench/groot_phases.py \
        --outdir $d --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps $N \
        --warmup 20 --timed 50 --rung $rung $RA"
    done
  done
  # T009 step 1 also asks for the repository's OWN benchmark, to compare per-component numbers.
  # Run as shipped: base checkpoint, the DROID demo sample (matches the embodiment tag the rest
  # of this campaign uses). If it fails on this device the error IS the result -- recorded,
  # not worked around by substituting another script.
  d=results/${DATE}_E3E4_groot_n17_benchmark_inference; mkdir -p "$d/raw"
  CK=$(ls -d ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/* | head -1)
  run "Isaac-GR00T scripts/deployment/benchmark_inference.py" /bin/bash -c \
    ". $GV/activate_orin_venv.sh >/dev/null 2>&1; cd /home/yanggon/00_VLA/Isaac-GR00T && \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_yanggon \
       TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon $GV/bin/python scripts/deployment/benchmark_inference.py \
       --model-path $CK --dataset-path demo_data/droid_sample \
       --embodiment-tag oxe_droid_relative_eef_relative_joint --num-iterations 20 --warmup 5 \
       > $PWD/$d/raw/benchmark_inference.log 2>&1"
  echo "  benchmark_inference exit recorded above; log in $d/raw/"
  echo "E3E4_SWEEP_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e6_groot_old)
  # Plan step 6: E6 for GR00T N1.5 and N1.6. No branch switch: their worktrees and the groot16
  # venv are how queue_e7_50w_groot.sh already runs them. Bytes do not change across rungs
  # (tf32 / compilation only), so one floor per version serves its eager and compiled rows.
  gate
  echo "E6_GROOT_OLD_START $(date -Is)"
  V16=$HOME/venvs/groot16
  for spec in "n16 /home/yanggon/00_VLA/Isaac-GR00T-n1d6 nvidia/GR00T-N1.6-3B new robocasa_panda_omron" \
              "n15 /home/yanggon/00_VLA/Isaac-GR00T-n1d5 nvidia/GR00T-N1.5-3B old oxe_droid"; do
    set -- $spec; v=$1; root=$2; ck=$3; api=$4; tag=$5
    d=results/${DATE}_E6_groot_$v
    if [ ! -s "$d/floor.csv" ]; then
      run "floors groot_$v" /bin/bash -c ". $V16/activate_orin_venv.sh >/dev/null 2>&1; \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 exec $V16/bin/python bench/floors_groot.py \
        --outdir $d --checkpoint $ck --version $v --api $api --embodiment-tag $tag \
        --data-config oxe_droid --gr00t-root $root --steps 4"
    fi
    [ -s "$d/floor.csv" ] || { echo "  NO floor for groot_$v -- rooflines skipped"; continue; }
    for r in eager_tf32 all_compile_autotune; do
      rd=results/${DATE}_E6_rungs/groot_${v}_${r}; mkdir -p "$rd"
      [ -s "$rd/roofline.csv" ] && continue
      cp "$d/floor.csv" "$d/floor_inputs.json" "$rd/"
      run "roofline groot_$v $r" $PY bench/roofline.py --outdir "$rd" --floor "$rd/floor.csv" \
          --ladder-row results/20260923_E4d_groot_${v}_ladder/raw/rows/${r}.json --model groot_$v
    done
  done
  echo "E6_GROOT_OLD_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e8)
  # Plan step 5, the owner's CORE scope: pi0 only (pi0.5's base checkpoint scores 0/30 and
  # cannot be validated). fp32 and the FULL bf16_both rung at N=10 on all four suites, 10 tasks
  # x 10 episodes; bf16_both at N in {4, 2} on libero_spatial. fp32 runs first on each suite as
  # the harness's own control: if it collapses to zero there, bf16 on that suite is skipped --
  # comparing against zero distinguishes nothing.
  gate
  echo "E8_START $(date -Is)"
  LPY=$HOME/venvs/libero/bin/python
  export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  # Is the FULL compiled rung usable in the closed loop? One-episode smoke, into scratch. If
  # not, the bf16 arm falls back to the casts-only prep and every result says so.
  BF="--rung bf16_both"; BFTAG=bf16both
  if ! $LPY bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_spatial --tasks 1 \
        --episodes 1 --outdir scratch/e8_smoke_rung > scratch/e8_smoke_rung.log 2>&1; then
    echo "  *** full bf16_both rung FAILED in the closed loop (scratch/e8_smoke_rung.log); the"
    echo "      bf16 arm uses the casts-only prep instead and is labelled bf16castonly"
    tail -5 scratch/e8_smoke_rung.log
    BF="--prep bf16_both"; BFTAG=bf16castonly
  fi
  for suite in libero_spatial libero_object libero_goal libero_10; do
    d=results/${DATE}_E8_pi0_${suite}_fp32_N10_settle10
    [ -s "$d/summary.json" ] || run "E8 $suite fp32 N=10" $LPY bench/libero_eval.py --model pi0 \
        --suite $suite --tasks 10 --episodes 10 --steps 10 --outdir "$d"
    ok=$($PY -c "import json;print(json.load(open('$d/summary.json'))['successes'])" 2>/dev/null || echo 0)
    if [ "${ok:-0}" -eq 0 ]; then
      echo "  *** fp32 control scored 0 on $suite -- bf16 arm SKIPPED there (nothing to compare)"
      continue
    fi
    d=results/${DATE}_E8_pi0_${suite}_${BFTAG}_N10_settle10
    [ -s "$d/summary.json" ] || run "E8 $suite $BFTAG N=10" $LPY bench/libero_eval.py --model pi0 \
        --suite $suite --tasks 10 --episodes 10 --steps 10 $BF --outdir "$d"
  done
  for N in 4 2; do
    d=results/${DATE}_E8_pi0_libero_spatial_${BFTAG}_N${N}_settle10
    [ -s "$d/summary.json" ] || run "E8 libero_spatial $BFTAG N=$N" $LPY bench/libero_eval.py \
        --model pi0 --suite libero_spatial --tasks 10 --episodes 10 --steps $N $BF --outdir "$d"
  done
  echo "E8_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e8_steplog)
  # The first settle10 arm (spatial, stock/"fp32", 63/100) ran before libero_eval.py wrote the
  # per-step raw log T006 requires. Same command again into a new directory: supplies that log
  # and doubles as a same-seed repeatability check of the success count.
  gate
  LPY=$HOME/venvs/libero/bin/python; export MUJOCO_GL=egl
  d=results/${DATE}_E8_pi0_libero_spatial_fp32_N10_settle10_steplog
  [ -s "$d/summary.json" ] || run "E8 libero_spatial fp32 N=10 (step-log re-run)" $LPY bench/libero_eval.py \
      --model pi0 --suite libero_spatial --tasks 10 --episodes 10 --steps 10 --outdir "$d"
  echo "E8_STEPLOG_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e12b_dram)
  # analysis/E12b_review.md owed question: measured DRAM bytes of the E12b prefix / expert vs
  # E6's analytic bytes vs the EMC-implied bytes (the 3.6x gap). One ncu capture, metrics only.
  gate
  run "E12b DRAM bytes (ncu, graphs_both prefix_alone + expert_alone)" \
      bash bench/run_ncu_e12b_dram.sh results/${DATE}_E12b_pi0_dram_bytes_MAXN
  echo "E12B_DRAM_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
e12_compiled_ref)
  # MAXN references for the capped session's step 0c, from the SAME script it will run there
  # (bench/e12_capped_compiled_stages.py): whole GPU and the confined green 8/8. Also proves the
  # script works before a reboot session depends on it.
  gate
  run "compiled stages alone MAXN whole GPU" $PY bench/e12_capped_compiled_stages.py --frames 60 \
      --outdir results/${DATE}_E12_pi0_compiled_stages_N10_MAXN
  run "compiled stages alone MAXN green 8/8" $PY bench/e12_capped_compiled_stages.py --frames 60 \
      --split 8/8 --outdir results/${DATE}_E12_pi0_compiled_stages_N10_MAXN_green8_8
  echo "E12_COMPILED_REF_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
groot_bench)
  # T009 step 1: NVIDIA's own benchmark_inference.py (PyTorch eager + torch.compile modes; no
  # TensorRT engine given, TRT is E10). The 2026-09-25 16:43 attempt failed on a missing
  # torchcodec (kept as results/20260925_E3E4_groot_n17_benchmark_inference). torchcodec 0.10.0a0
  # was then installed --no-deps from the repo's own Orin wheel (env/INSTALL_LOG.md); a new
  # directory, never the failed one.
  gate
  GV=$HOME/venvs/groot
  d=results/${DATE}_E3E4_groot_n17_benchmark_inference_torchcodec
  [ -e "$d" ] && { echo "REFUSING: $d exists"; exit 2; }
  mkdir -p "$d/raw"
  CK=$(ls -d ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/* | head -1)
  CMD="scripts/deployment/benchmark_inference.py --model-path $CK --dataset-path demo_data/droid_sample --embodiment-tag oxe_droid_relative_eef_relative_joint --num-iterations 20 --warmup 5"
  GS=$(sudo -n nvpmodel -q 2>/dev/null | head -1); GF=$(cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq 2>/dev/null)
  run "Isaac-GR00T benchmark_inference.py (torchcodec)" /bin/bash -c \
    ". $GV/activate_orin_venv.sh >/dev/null 2>&1; cd /home/yanggon/00_VLA/Isaac-GR00T && \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_yanggon \
       TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon $GV/bin/python $CMD \
       > $PWD/$d/raw/benchmark_inference.log 2>&1"
  python3 - "$d" "$CMD" "$GS" "$GF" <<'PY'
import json, sys, time, subprocess, hashlib, pathlib
d, cmd, gs, gf = sys.argv[1:5]
log = pathlib.Path(d, "raw/benchmark_inference.log")
txt = log.read_text(errors="replace")
whl = pathlib.Path("/home/yanggon/00_VLA/Isaac-GR00T/scripts/deployment/orin/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl")
json.dump({
  "experiment": "E3/E4 GR00T N1.7 -- NVIDIA's own scripts/deployment/benchmark_inference.py (T009 step 1 comparison), re-run with torchcodec",
  "status": "completed" if "Traceback" not in txt else "FAILED -- see raw/benchmark_inference.log",
  "gr00t_commit": subprocess.run(["git", "-C", "/home/yanggon/00_VLA/Isaac-GR00T", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
  "bench_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
  "checkpoint": "nvidia/GR00T-N1.7-3B", "embodiment_tag": "oxe_droid_relative_eef_relative_joint",
  "dataset": "demo_data/droid_sample (git-lfs pulled)", "power_mode": gs, "gpu_cur_freq_hz": gf,
  "harness_change": "torchcodec 0.10.0a0 installed --no-deps from Isaac-GR00T's Orin wheel (sha256 "
                    + hashlib.sha256(whl.read_bytes()).hexdigest() + "); uv pip freeze before/after differs only by that line",
  "supersedes_failure": "results/20260925_E3E4_groot_n17_benchmark_inference (ImportError: torchcodec)",
  "command": cmd, "cwd": "/home/yanggon/00_VLA/Isaac-GR00T", "log": "raw/benchmark_inference.log",
  "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(pathlib.Path(d, "meta.json"), "w"), indent=2)
PY
  echo "GROOT_BENCH_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
chain)
  # Everything after e6_rungs, in plan order, one GPU job at a time. Waits for a running
  # e6_rungs to finish first rather than overlapping it.
  # ...or until that stage's process is gone: a stage that died without its DONE marker must
  # not leave this chain waiting forever while the watchdog reports NOTHING RUNNING.
  until grep -q "E6_RUNGS_DONE" scratch/queues/queue_prereboot_e6_rungs.log 2>/dev/null \
        || ! pgrep -f "[q]ueue_prereboot[.]e6_rungs" >/dev/null; do sleep 20; done
  grep -q "E6_RUNGS_DONE" scratch/queues/queue_prereboot_e6_rungs.log \
    || echo "  *** e6_rungs ended WITHOUT its DONE marker -- continuing, check its log"
  for st in e3e4_sweep e6_groot_old e8; do
    echo "########## $(date +%H:%M) CHAIN stage $st"
    QP_SNAPSHOT= bash bench/queues/queue_prereboot.sh $st > scratch/queues/queue_prereboot_$st.log 2>&1
    echo "EXIT=$? CHAIN stage $st"
  done
  echo "PREREBOOT_CHAIN_DONE $(date -Is)"
  ;;
# ---------------------------------------------------------------------------------------
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
