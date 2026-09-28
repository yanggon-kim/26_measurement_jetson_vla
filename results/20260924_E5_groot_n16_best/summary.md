# 20260924_E5_groot_n16_best

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:47:42Z, power mode NV Power Mode: MAXN, model groot_n16
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n16_best --tag best --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (6 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 64.608 | 0.4251 | 0.9049 | 0.4251 | 4 | 2 | 0.0447 | 8.0 |
| 61.888 | 0.4008 | 0.9006 | 0.4008 | 4 | 2 | 0.0443 | 9.0 |
| 0.001 | 0.0716 | 0.001 | 0.0716 | 2 | 1 | 73.007 | 1392.0 |
| 1.376 | 6.0597 | 0.1244 | 6.0597 | 2 | 1 | 42.6427 | 382.0 |
| 1.376 | 12.2208 | 0.2786 | 12.2208 | 2 | 1 | 31.6442 | 249.0 |
|  |  | 0.138 | 47.4327 | 1 | 1 | 296.2231 | 2063.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
