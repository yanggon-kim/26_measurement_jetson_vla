# 20260924_E5b_pi05_best

* **What:** see meta.json

Headline — first rows of `stall_attribution.csv` (238 rows):

| model | mode | region | kernel | duration_us | launches | sm_throughput_pct | launch_bound_s1 |
|---|---|---|---|---|---|---|---|
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |
| pi05 | best | ae_step_0 | triton_per_fused_sum_0 | 7.42 | 1 | 0.22 | yes |

**Anything odd:** _backfill: {'backfilled': True, 'backfilled_at': '2026-09-25T13:42:16Z', 'by': 'bench/backfill_meta.py', 'why': 'this directory was written before the §5 meta.json requirement was enforced on every experiment. T; raw_files_note: 2026-09-28 completeness audit: raw_files listing added (sha256 of every file in the directory)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*

**2026-09-28 — docs/03 §0 file set completed** (`bench/e5b_static.py`, CPU only, from this directory's own ncu reports): 17 distinct kernel symbols in `sass/targets.csv` (12 past the §1 launch-bound bar get hot loop, pseudocode and latency budget), per-symbol SASS under `raw/<kernel>_sm87.*`, `ncu/parsed.json`, `ncu/*_raw.csv`; 15 items not produced, each with its reason in meta.json `not_produced`. The earlier `inventory/` files were per SHORT name and are superseded (meta.json `e5b_static`).
