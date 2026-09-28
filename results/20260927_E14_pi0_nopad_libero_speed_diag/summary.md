# 20260927_E14_pi0_nopad_libero_speed_diag

* **What:** E8 LIBERO closed loop (T006 step 2)
* **Conditions:** date 2026-09-27T20:15:42Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/libero_eval.py --model pi0 --rung bf16_both --suite libero_spatial --tasks 2 --episodes 3 --steps 10 --seed-noise --prefix-variant nopad --outdir results/20260927_E14_pi0_nopad_libero_speed_diag/nopad_default`
* **Sub-runs:** nopad_default, nopad_static, pad_static

Headline — first rows of `nopad_default/episodes.csv` (6 rows):

| task | task_name | episode | success | steps | noise_seed |
|---|---|---|---|---|---|
| 0 | pick up the black bowl … | 0 | 1 | 72 | 7919000000 |
| 0 | pick up the black bowl … | 1 | 0 | 280 | 7919007919 |
| 0 | pick up the black bowl … | 2 | 1 | 151 | 7919015838 |
| 1 | pick up the black bowl … | 0 | 1 | 111 | 7926919000 |
| 1 | pick up the black bowl … | 1 | 1 | 106 | 7926926919 |
| 1 | pick up the black bowl … | 2 | 1 | 111 | 7926934838 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
