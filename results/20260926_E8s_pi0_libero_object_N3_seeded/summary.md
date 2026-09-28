# 20260926_E8s_pi0_libero_object_N3_seeded

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-27T12:03:19Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_object --tasks 10 --episodes 10 --steps 3 --seed-noise --outdir results/20260926_E8s_pi0_libero_object_N3_seeded`

Headline — first rows of `episodes.csv` (100 rows):

| task | task_name | episode | success | steps | noise_seed |
|---|---|---|---|---|---|
| 0 | pick up the alphabet so… | 0 | 1 | 127 | 15838000000 |
| 0 | pick up the alphabet so… | 1 | 1 | 139 | 15838007919 |
| 0 | pick up the alphabet so… | 2 | 1 | 122 | 15838015838 |
| 0 | pick up the alphabet so… | 3 | 1 | 142 | 15838023757 |
| 0 | pick up the alphabet so… | 4 | 0 | 280 | 15838031676 |
| 0 | pick up the alphabet so… | 5 | 1 | 132 | 15838039595 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
