# 20260928_E5_pi05_best_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-28T20:05:17Z, power mode NV Power Mode: MAXN, model pi05, rung / mode bf16_both_graphs
* **Command:** `bench/parse_nsys.py results/20260928_E5_pi05_best_N4 --tag best_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 80.864 | 0.4727 | 0.855 | 0.4727 | 4 | 2 | 0.0801 | 8.0 |
| 0.096 | 38.4156 | 0.4486 | 38.4156 | 8 | 4 | 47.2221 | 401.25 |
| 0.001 | 0.6474 | 0.0113 | 0.6474 | 6 | 3 | 56.7598 | 837.0 |
| 0.001 | 0.244 | 0.001 | 0.244 | 2 | 1 | 234.7077 | 350.0 |
|  |  | 0.1237 | 95.7454 | 1 | 1 | 678.1614 | 2832.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
