#!/usr/bin/env bash
# E12 Step 3, two processes: the first configuration in which the co-run question can be asked.
#
# The in-process pipeline established two blockers (reports/E12_pi0_corun.md §3, §4):
# pi0 eager is CPU-issue-bound (issue/wall = 1.0000, so nothing overlaps and GPU contention
# cannot arise), and the compiled bf16_both rung corrupts when pipelined across two streams.
# Two processes clears both: two interpreters issue on two cores, and each process has ONE
# sequential call stream so CUDAGraphTrees never sees two. That makes the COMPILED rung --
# 372 kernels/step, 47 % idle, the one that actually loads the GPU -- usable.
#
# Validation already in hand: the compiled prefix measures 250.36 ms here against E7's
# 251.28 ms, a 0.37 % match through a different harness (non-negotiable 3's reconciliation).
#
# Isolated and co-run are separate PROCESSES by construction, so "same session" (§3) is met by
# running them back to back in one power mode and one thermal state, and by bracketing:
# isolated -> co-run -> isolated, with the drift reported.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
SEC="${SEC:-60}"
RUNG="${RUNG:-bf16_both}"
N="${N:-10}"
OUT=results/$(date +%Y%m%d)_E12_pi0_2proc_N${N}_MAXN
R=/tmp/e12_ready; rm -rf "$R"; mkdir -p "$R" "$OUT/raw"
export TRITON_CACHE_DIR=/tmp/triton_yanggon TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

m=$(sudo -n nvpmodel -q 2>/dev/null | head -1)
case "$m" in *MAXN*) echo "power mode: $m" ;;
  *) echo "REFUSING: power mode is '$m', expected MAXN"; exit 2 ;; esac

echo "E12_2PROC_START $(date -Is)   rung=$RUNG N=$N seconds=$SEC"

run1() {   # $1 role  $2 tag   -- isolated: no --peer, so it times immediately after warm-up
  echo "########## $(date +%H:%M) isolated $1 ($2)"
  $PY bench/corun_2proc.py --role "$1" --rung "$RUNG" --steps "$N" --seconds "$SEC" \
      --out "$OUT/${1}_${2}.json" > "$OUT/raw/${1}_${2}.log" 2>&1
  echo "EXIT=$? isolated $1 ($2)"
}

# --- bracket 1: isolated
run1 prefix iso_pre
run1 expert iso_pre

# --- co-run: both at once, each waiting for the other's ready-file before timing
echo "########## $(date +%H:%M) CO-RUN (both processes, handshake on ready-files)"
$PY bench/corun_2proc.py --role prefix --rung "$RUNG" --steps "$N" --seconds "$SEC" \
    --ready "$R/p.ready" --peer "$R/e.ready" --out "$OUT/prefix_corun.json" \
    > "$OUT/raw/prefix_corun.log" 2>&1 &
pp=$!
$PY bench/corun_2proc.py --role expert --rung "$RUNG" --steps "$N" --seconds "$SEC" \
    --ready "$R/e.ready" --peer "$R/p.ready" --out "$OUT/expert_corun.json" \
    > "$OUT/raw/expert_corun.log" 2>&1 &
pe=$!
wait $pp; echo "EXIT=$? corun prefix"
wait $pe; echo "EXIT=$? corun expert"

# --- bracket 2: isolated again, so drift is visible rather than assumed
run1 prefix iso_post
run1 expert iso_post

$PY bench/e12_2proc_ratios.py --dir "$OUT" || echo "  RATIOS FAILED"
echo "E12_2PROC_DONE $(date -Is)"
