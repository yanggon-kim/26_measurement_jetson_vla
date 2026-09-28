# 20260924_E5_pi0_best_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:30:29Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `bench/parse_nsys.py results/20260924_E5_pi0_best_N10 --tag best_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 62.112 | 0.4361 | 0.8779 | 0.4361 | 4 | 2 | 0.0607 | 8.0 |
| 0.096 | 93.043 | 0.4714 | 93.043 | 20 | 10 | 104.3424 | 372.7 |
| 0.001 | 0.6271 | 0.0111 | 0.6271 | 6 | 3 | 55.6713 | 837.0 |
| 1.376 | 0.8839 | 0.0046 | 0.8839 | 2 | 1 | 191.3742 | 346.0 |
|  |  | 0.2262 | 205.6291 | 1 | 1 | 703.444 | 4952.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
