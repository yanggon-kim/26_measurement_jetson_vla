# 20260924_E5_groot_n16_eager

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:46:03Z, power mode NV Power Mode: MAXN, model groot_n16
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n16_eager --tag eager --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (7 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 54.688 | 0.3856 | 0.899 | 0.3856 | 4 | 2 | 0.0433 | 8.0 |
| 15.456 | 24.0796 | 0.3613 | 24.0796 | 54 | 27 | 42.5597 | 522.0 |
| 47.104 | 0.3713 | 0.8975 | 0.3713 | 4 | 2 | 0.0424 | 9.0 |
| 67.264 | 214.5368 | 0.7224 | 214.5368 | 2 | 1 | 82.452 | 3103.0 |
| 16.8 | 29.1631 | 0.3874 | 29.1631 | 2 | 1 | 46.1132 | 586.0 |
| 39.456 | 38.8568 | 0.4466 | 38.8568 | 2 | 1 | 48.1572 | 983.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
