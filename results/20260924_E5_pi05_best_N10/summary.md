# 20260924_E5_pi05_best_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:41:06Z, power mode NV Power Mode: MAXN, model pi05, rung / mode bf16_both_graphs
* **Command:** `bench/parse_nsys.py results/20260924_E5_pi05_best_N10 --tag best_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 80.736 | 0.48 | 0.8575 | 0.48 | 4 | 2 | 0.0797 | 8.0 |
| 0.096 | 94.9711 | 0.4453 | 94.9711 | 20 | 10 | 118.3044 | 401.7 |
| 0.001 | 0.6223 | 0.0111 | 0.6223 | 6 | 3 | 55.4964 | 837.0 |
| 0.001 | 0.2392 | 0.0011 | 0.2392 | 2 | 1 | 226.5775 | 350.0 |
|  |  | 0.2062 | 208.2209 | 1 | 1 | 801.5474 | 5244.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
