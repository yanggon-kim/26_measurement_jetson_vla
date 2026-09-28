# 20260924_E5_groot_n17_eager

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:43:47Z, power mode NV Power Mode: MAXN, model groot_n17
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n17_eager --tag eager --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (8 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 47.968 | 70.5119 | 0.6572 | 70.5119 | 52 | 26 | 36.7796 | 1364.0 |
| 19.328 | 0.0185 | 0.352 | 0.0185 | 2 | 1 | 0.0341 | 2.0 |
| 44.672 | 0.4932 | 0.9542 | 0.4932 | 4 | 2 | 0.0237 | 11.0 |
| 47.84 | 70.499 | 0.6572 | 70.499 | 52 | 26 | 36.7764 | 1364.0 |
| 69.856 | 223.4031 | 0.7153 | 223.4031 | 2 | 1 | 88.8997 | 3168.0 |
| 1.408 | 36.36 | 0.4132 | 36.36 | 2 | 1 | 51.6419 | 921.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
