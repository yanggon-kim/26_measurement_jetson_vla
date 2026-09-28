# 20260927_E8s_pi0_libero_spatial_N10_seeded_maskedcam

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-27T21:10:41Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_spatial --steps 10 --seed-noise --static-shapes --empty-camera masked --tasks 10 --episodes 10 --prefix-variant pad --outdir results/20260927_E8s_pi0_liber`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps | noise_seed |
|---|---|---|---|---|---|
| 0 | pick up the black bowl … | 0 | 0 | 280 | 7919000000 |
| 0 | pick up the black bowl … | 1 | 1 | 103 | 7919007919 |
| 0 | pick up the black bowl … | 2 | 0 | 280 | 7919015838 |
| 0 | pick up the black bowl … | 3 | 0 | 280 | 7919023757 |
| 0 | pick up the black bowl … | 4 | 0 | 280 | 7919031676 |
| 0 | pick up the black bowl … | 5 | 1 | 68 | 7919039595 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
