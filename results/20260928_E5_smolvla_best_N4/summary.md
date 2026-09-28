# 20260928_E5_smolvla_best_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-28T20:06:46Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode vlm_graphs
* **Command:** `bench/parse_nsys.py results/20260928_E5_smolvla_best_N4 --tag best_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 74.08 | 0.4664 | 0.9555 | 0.4664 | 4 | 2 | 0.0217 | 8.0 |
| 0.096 | 8.0311 | 0.3596 | 8.0311 | 8 | 4 | 14.3024 | 355.25 |
| 0.001 | 1.5614 | 0.0496 | 1.5614 | 4 | 2 | 29.9338 | 288.0 |
| 0.096 | 0.3455 | 0.0467 | 0.3455 | 2 | 1 | 7.0554 | 341.0 |
|  |  | 0.2609 | 36.5266 | 1 | 1 | 103.486 | 2089.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
