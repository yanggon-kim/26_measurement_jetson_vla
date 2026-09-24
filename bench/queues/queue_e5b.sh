#!/usr/bin/env bash
# E5b / T007 — instruction-level analysis of the dominant action-expert kernels.
#
# The targets are NOT chosen by hand and NOT found by exploratory captures. C1 already
# measured every kernel of ae_step_0 for all three models in both modes, with duration and
# SM throughput, so bench/pick_e5b_targets.py applies the document's own two rules to that
# committed data:
#
#   docs/03 §1  skip a kernel below 5 % SM throughput whose duration is comparable to the
#               launch gap (62 us, measured in E5) -- it is launch-bound and its
#               instruction mix explains nothing
#   docs/03 §5  the chosen targets must cover >= 60 % of ae_step kernel time
#
# Every model clears the 60 % bar with between one and eight kernels. pi0.5's compiled
# expert needs exactly ONE: 96.4 % of its captured kernel time is a single cuBLAS
# internal::kernel, a GEMV, at 13.2 % SM throughput.
#
# Never two profilers at once (owner, 2026-09-23). run_ncu_e5b.sh refuses to start while
# another is live, and these run strictly in sequence.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
export DATE=20260924
PY=$HOME/venvs/pi0/bin/python

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; pkill -f "[n]sys profile"; sleep 2; true' 2>/dev/null
}

echo "E5B_START $(date -Is)"
sweep

# The picker writes one `bash bench/run_ncu_e5b.sh ...` line per target. Generating them
# at run time rather than pasting them here means the queue follows the committed C1 data:
# if a C1 capture is corrected, the targets change with it instead of silently not.
TMP=scratch/queues/e5b_targets.sh
mkdir -p scratch/queues
$PY bench/pick_e5b_targets.py --emit-queue > "$TMP" 2> scratch/queues/e5b_targets.err
echo "  $(wc -l < "$TMP") capture(s) planned"
cat scratch/queues/e5b_targets.err >&2 || true

while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "########## $(date +%H:%M) $line"
  sweep
  eval "$line"
  echo "EXIT=$?"
done < "$TMP"

# One analysis pass over everything captured, and the report.
$PY bench/analyze_e5b.py --glob "results/${DATE}_E5b_*" --md reports/E5b_summary.md
echo "E5B_DONE $(date -Is)"
