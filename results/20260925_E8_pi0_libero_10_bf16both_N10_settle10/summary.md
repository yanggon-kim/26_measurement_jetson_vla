# 20260925_E8_pi0_libero_10_bf16both_N10_settle10

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-26T04:21:25Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --suite libero_10 --tasks 10 --episodes 10 --steps 10 --rung bf16_both --outdir results/20260925_E8_pi0_libero_10_bf16both_N10_settle10`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps |
|---|---|---|---|---|
| 0 | put both the alphabet s… | 0 | 0 | 520 |
| 0 | put both the alphabet s… | 1 | 0 | 520 |
| 0 | put both the alphabet s… | 2 | 0 | 520 |
| 0 | put both the alphabet s… | 3 | 0 | 520 |
| 0 | put both the alphabet s… | 4 | 0 | 520 |
| 0 | put both the alphabet s… | 5 | 0 | 520 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
