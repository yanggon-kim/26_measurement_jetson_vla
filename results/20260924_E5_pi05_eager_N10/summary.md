# 20260924_E5_pi05_eager_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:35:52Z, power mode NV Power Mode: MAXN, model pi05
* **Command:** `bench/parse_nsys.py results/20260924_E5_pi05_eager_N10 --tag eager_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 48.672 | 529.3809 | 0.1765 | 529.3809 | 44 | 22 | 2470.0824 | 10852.0 |
| 51.232 | 580.1883 | 0.6429 | 580.1883 | 20 | 10 | 322.2665 | 1102.7 |
| 1.344 | 2.6316 | 0.0108 | 2.6316 | 6 | 3 | 239.9558 | 1650.0 |
| 1.344 | 2.3901 | 0.0011 | 2.3901 | 2 | 1 | 2179.7444 | 879.0 |
|  |  | 0.1773 | 1182.2264 | 1 | 1 | 5484.6367 | 13593.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
