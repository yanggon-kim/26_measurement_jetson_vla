# 20260927_E7_power_modes_compare

* **What:** T004 step 4b: E7 across power modes (MAXN, MODE_50W, MODE_30W, MODE_15W) -- derived comparison, no new measurement
* **Conditions:** date 2026-09-28T05:50:00Z
* **Command:** `python3 bench/e7_capped_compare.py --out results/20260927_E7_power_modes_compare`

Headline — first rows of `e7_power_modes.csv` (60 rows):

| model | rung | N | matmul | phase | MAXN_ms | MAXN_mJ_above_idle | MAXN_src |
|---|---|---|---|---|---|---|---|
| groot_n15 | all_compile_autotune | 4 | high | prefix | 98.08 | 5544.0 | w |
| groot_n15 | all_compile_autotune | 4 | high | ae | 60.24 | 3020.5 | w |
| groot_n15 | all_compile_autotune | 4 | high | full | 159.65 | 8709.4 | w |
| groot_n15 | best | 4 | high | prefix |  |  |  |
| groot_n15 | best | 4 | high | ae |  |  |  |
| groot_n15 | best | 4 | high | full |  |  |  |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
