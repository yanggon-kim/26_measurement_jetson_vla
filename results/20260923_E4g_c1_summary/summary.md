# 20260923_E4g_c1_summary

* **What:** see meta.json

Headline — first rows of `c1_bounds.csv` (36 rows):

| model | mode | region | n_kernels | total_us | sm_pct | mem_pct | occupancy_pct |
|---|---|---|---|---|---|---|---|
| groot_n15 | eager | vision | 40 | 3268.41 | 49.35 | 70.96 | 48.82 |
| groot_n15 | best | vision | 40 | 3240.73 | 49.45 | 71.5 | 48.84 |
| groot_n15 | eager | vlm_prefix | 40 | 1967.17 | 37.11 | 51.81 | 63.12 |
| groot_n15 | best | vlm_prefix | 40 | 9767.8 | 50.2 | 83.47 | 21.94 |
| groot_n15 | eager | ae_total | 40 | 6648.27 | 54.19 | 71.65 | 30.62 |
| groot_n15 | best | ae_total | 40 | 13386.13 | 63.99 | 76.01 | 22.72 |

**Anything odd:** _backfill: {'backfilled': True, 'backfilled_at': '2026-09-25T13:42:16Z', 'by': 'bench/backfill_meta.py', 'why': 'this directory was written before the §5 meta.json requirement was enforced on every experiment. T

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
