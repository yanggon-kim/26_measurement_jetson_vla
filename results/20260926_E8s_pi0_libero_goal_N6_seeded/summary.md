# 20260926_E8s_pi0_libero_goal_N6_seeded

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-27T14:01:16Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_goal --tasks 10 --episodes 10 --steps 6 --seed-noise --outdir results/20260926_E8s_pi0_libero_goal_N6_seeded`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps | noise_seed |
|---|---|---|---|---|---|
| 0 | open the middle drawer … | 0 | 1 | 127 | 23757000000 |
| 0 | open the middle drawer … | 1 | 1 | 139 | 23757007919 |
| 0 | open the middle drawer … | 2 | 1 | 127 | 23757015838 |
| 0 | open the middle drawer … | 3 | 1 | 136 | 23757023757 |
| 0 | open the middle drawer … | 4 | 1 | 123 | 23757031676 |
| 0 | open the middle drawer … | 5 | 1 | 137 | 23757039595 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
