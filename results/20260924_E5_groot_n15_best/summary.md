# 20260924_E5_groot_n15_best

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:50:20Z, power mode NV Power Mode: MAXN, model groot_n15
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n15_best --tag best --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (6 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 42.08 | 0.3811 | 0.7294 | 0.3811 | 4 | 2 | 0.1414 | 8.0 |
| 39.84 | 0.3488 | 0.7114 | 0.3488 | 4 | 2 | 0.1415 | 9.0 |
| 0.001 | 0.0492 | 0.0008 | 0.0492 | 2 | 1 | 60.0366 | 760.0 |
| 1.376 | 0.4747 | 0.0149 | 0.4747 | 2 | 1 | 31.3365 | 286.0 |
| 1.376 | 6.264 | 0.0968 | 6.264 | 2 | 1 | 58.4366 | 189.0 |
|  |  | 0.0881 | 29.0395 | 1 | 1 | 300.6277 | 1265.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
