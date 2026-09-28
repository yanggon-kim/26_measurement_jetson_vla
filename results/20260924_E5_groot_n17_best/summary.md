# 20260924_E5_groot_n17_best

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:45:19Z, power mode NV Power Mode: MAXN, model groot_n17
* **Command:** `bench/parse_nsys.py results/20260924_E5_groot_n17_best --tag best --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (8 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 51.488 | 63.7327 | 0.6646 | 63.7327 | 52 | 26 | 32.169 | 928.0 |
| 19.456 | 0.0191 | 0.3528 | 0.0191 | 2 | 1 | 0.035 | 2.0 |
| 45.824 | 0.4914 | 0.9533 | 0.4914 | 4 | 2 | 0.0241 | 11.0 |
| 51.488 | 63.7326 | 0.6646 | 63.7326 | 52 | 26 | 32.1658 | 928.0 |
| 0.001 | 0.0676 | 0.0008 | 0.0676 | 2 | 1 | 80.1601 | 1631.0 |
| 1.408 | 37.6892 | 0.4216 | 37.6892 | 2 | 1 | 51.7077 | 921.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
