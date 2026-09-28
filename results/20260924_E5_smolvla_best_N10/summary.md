# 20260924_E5_smolvla_best_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:42:58Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode vlm_graphs
* **Command:** `bench/parse_nsys.py results/20260924_E5_smolvla_best_N10 --tag best_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 72.928 | 0.4664 | 0.9561 | 0.4664 | 4 | 2 | 0.0214 | 8.0 |
| 0.096 | 21.7764 | 0.3795 | 21.7764 | 20 | 10 | 35.6023 | 355.7 |
| 0.001 | 1.6395 | 0.0518 | 1.6395 | 4 | 2 | 30.0293 | 288.0 |
| 0.096 | 0.3614 | 0.049 | 0.3614 | 2 | 1 | 7.0087 | 341.0 |
|  |  | 0.3065 | 64.6091 | 1 | 1 | 146.1926 | 4225.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
