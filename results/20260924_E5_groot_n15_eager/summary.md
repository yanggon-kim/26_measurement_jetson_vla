# 20260924_E5_groot_n15_eager

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:48:54Z, power mode NV Power Mode: MAXN, model groot_n15
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n15_eager --tag eager --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (6 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 37.12 | 0.2917 | 0.6756 | 0.2917 | 4 | 2 | 0.1401 | 8.0 |
| 25.344 | 0.2691 | 0.6551 | 0.2691 | 4 | 2 | 0.1417 | 9.0 |
| 44.608 | 81.869 | 0.5258 | 81.869 | 2 | 1 | 73.8218 | 1982.0 |
| 1.376 | 6.0062 | 0.1078 | 6.0062 | 2 | 1 | 49.6888 | 709.0 |
| 1.376 | 6.6435 | 0.0655 | 6.6435 | 2 | 1 | 94.7279 | 754.0 |
|  |  | 0.3169 | 202.9262 | 1 | 1 | 437.4537 | 3475.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
