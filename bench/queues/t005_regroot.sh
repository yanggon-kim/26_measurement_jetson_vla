#!/usr/bin/env bash
# T005 review follow-up (2026-10-24, review finding 13): re-measure the GR00T candidates that were excluded only by a
# transient disturbance, and the N1.5 linear_mpad variants stage 2 dropped on a premise the review showed to be wrong
# for N1.5 (padding speeds up its unused lm_head).
#   N1.7: ae_at, ae_at_mpad          (stage-2 rows were unstable: p95 363 / 371 ms; never re-run)
#   N1.5: ae_graphs_mpad_autotune, ae_graphs_mpad  (excluded only as "throttled": one oc2 event, rel L2 within bound)
#         ae_at_mpad, llm_graphs_ae_at_mpad        (padded variants not tried for N1.5 in stage 2)
set -u
D=${1:?date}
R=$(cd "$(dirname "$0")/../.." && pwd)
cd "$R"
source env/thor_env.sh
echo "##### $(date -u +%H:%M:%S) groot_n17 re-measure"
~/venvs/groot17/bin/python bench/opt_ladder.py --model groot_n17 --out "$R/results/${D}_E4i_groot_n17_rerun2" --rungs ae_at,ae_at_mpad
echo "##### $(date -u +%H:%M:%S) groot_n15 re-measure"
~/venvs/groot15/bin/python bench/opt_ladder.py --model groot_n15 --out "$R/results/${D}_E4i_groot_n15_rerun" \
    --rungs ae_graphs_mpad_autotune,ae_graphs_mpad,ae_at_mpad,llm_graphs_ae_at_mpad
echo "##### $(date -u +%H:%M:%S) re-measure done"
