# 20260925_E8_pi0_libero_spatial_fp32_N10

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-25T22:11:38Z, power mode NV Power Mode: MAXN, model pi0
* **Command:** `bench/libero_eval.py --model pi0 --suite libero_spatial --tasks 10 --episodes 10 --steps 10 --outdir results/20260925_E8_pi0_libero_spatial_fp32_N10`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps |
|---|---|---|---|---|
| 0 | pick up the black bowl … | 0 | 1 | 151 |
| 0 | pick up the black bowl … | 1 | 0 | 280 |
| 0 | pick up the black bowl … | 2 | 0 | 280 |
| 0 | pick up the black bowl … | 3 | 1 | 156 |
| 0 | pick up the black bowl … | 4 | 0 | 280 |
| 0 | pick up the black bowl … | 5 | 0 | 280 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
