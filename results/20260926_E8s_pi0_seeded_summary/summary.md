# 20260926_E8s_pi0_seeded_summary

* **What:** E8s (T016 step 2) seeded LIBERO N sweep -- derived analysis, no new measurement
* **Conditions:** date 2026-09-27T14:37:32Z, power mode MAXN (per input arm meta.json), model pi0 lerobot/pi0_libero_finetuned_v044, rung / mode bf16_both
* **Command:** `python3 bench/e8s_analyze.py results/20260926_E8s_pi0_seeded_summary`

Headline — first rows of `e8s_table.csv` (25 rows):

| suite | N | tag | dir | episodes | successes | rate | wilson_lo |
|---|---|---|---|---|---|---|---|
| libero_spatial | 1 |  | 20260926_E8s_pi0_libero… | 100 | 85 | 0.85 | 0.7671644033458782 |
| libero_spatial | 2 |  | 20260926_E8s_pi0_libero… | 100 | 80 | 0.8 | 0.7111708336357507 |
| libero_spatial | 3 |  | 20260926_E8s_pi0_libero… | 100 | 73 | 0.73 | 0.6356788315330401 |
| libero_spatial | 4 |  | 20260926_E8s_pi0_libero… | 100 | 76 | 0.76 | 0.667676635719002 |
| libero_spatial | 6 |  | 20260926_E8s_pi0_libero… | 100 | 70 | 0.7 | 0.6041514528775586 |
| libero_spatial | 10 |  | 20260926_E8s_pi0_libero… | 100 | 71 | 0.71 | 0.6146110874554646 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
