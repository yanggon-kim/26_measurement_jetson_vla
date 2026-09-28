# 20260922_E5_pi0_eager_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-24T08:37:33Z, power mode NV Power Mode: MAXN, model pi0
* **Command:** `bench/parse_nsys.py results/20260922_E5_pi0_eager_N4 --tag eager_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 54.56 | 225.9362 | 0.4524 | 225.9362 | 20 | 10 | 273.4355 | 3897.0 |
| 62.4 | 289.671 | 0.8427 | 289.671 | 8 | 4 | 54.0829 | 968.25 |
| 1.344 | 2.7967 | 0.0115 | 2.7967 | 6 | 3 | 240.0573 | 1650.0 |
| 1.376 | 3.79 | 0.016 | 3.79 | 2 | 1 | 232.4468 | 939.0 |
|  |  | 0.3649 | 605.4732 | 1 | 1 | 1053.8168 | 6504.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
