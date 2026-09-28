# 20260925_E8_pi0_libero_goal_fp32_N10_settle10

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-26T02:25:42Z, power mode NV Power Mode: MAXN, model pi0
* **Command:** `bench/libero_eval.py --model pi0 --suite libero_goal --tasks 10 --episodes 10 --steps 10 --outdir results/20260925_E8_pi0_libero_goal_fp32_N10_settle10`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps |
|---|---|---|---|---|
| 0 | open the middle drawer … | 0 | 1 | 124 |
| 0 | open the middle drawer … | 1 | 1 | 122 |
| 0 | open the middle drawer … | 2 | 1 | 127 |
| 0 | open the middle drawer … | 3 | 1 | 130 |
| 0 | open the middle drawer … | 4 | 1 | 116 |
| 0 | open the middle drawer … | 5 | 1 | 125 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
