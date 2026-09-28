# 20260924_E5_pi0_best_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:25:55Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/parse_nsys.py results/20260924_E5_pi0_best_N4 --tag best_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 61.152 | 0.4461 | 0.8792 | 0.4461 | 4 | 2 | 0.0613 | 8.0 |
| 0.096 | 37.3074 | 0.472 | 37.3074 | 8 | 4 | 41.7253 | 372.25 |
| 0.001 | 0.6177 | 0.0109 | 0.6177 | 6 | 3 | 55.9724 | 837.0 |
| 1.376 | 0.882 | 0.0046 | 0.882 | 2 | 1 | 191.3921 | 346.0 |
|  |  | 0.1386 | 93.1476 | 1 | 1 | 578.854 | 2714.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
