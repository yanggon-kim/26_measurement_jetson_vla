# 20260926_E8s_pi0_libero_10_N2_seeded

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-27T07:41:54Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_10 --tasks 10 --episodes 10 --steps 2 --seed-noise --outdir results/20260926_E8s_pi0_libero_10_N2_seeded`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps | noise_seed |
|---|---|---|---|---|---|
| 0 | put both the alphabet s… | 0 | 1 | 248 | 31676000000 |
| 0 | put both the alphabet s… | 1 | 0 | 520 | 31676007919 |
| 0 | put both the alphabet s… | 2 | 0 | 520 | 31676015838 |
| 0 | put both the alphabet s… | 3 | 1 | 273 | 31676023757 |
| 0 | put both the alphabet s… | 4 | 0 | 520 | 31676031676 |
| 0 | put both the alphabet s… | 5 | 0 | 520 | 31676039595 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
