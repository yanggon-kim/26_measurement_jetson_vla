#!/usr/bin/env bash
# T005 Part A: the published characterisation's configuration (eager + TF32, the model's own N) and its fp32 control,
# for pi0, pi0.5 and SmolVLA (reference/orin/bench/paper_repro.py's two arms), on the same harness as the ladder.
# Each arm in its own process; actions saved (a cross-process repeat of the ladder's eager rungs).
# Usage: bash bench/queues/t005_part_a.sh <YYYYMMDD>
set -u
D=${1:?date}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
for m in pi0 pi05 smolvla; do
  echo "##### $(date -u +%H:%M:%S) $m Part A"
  ~/venvs/lerobot/bin/python bench/opt_ladder.py --model $m --out results/${D}_E4a_${m}_paper_config \
      --rungs eager_tf32,eager_fp32
done
echo "##### $(date -u +%H:%M:%S) done"
