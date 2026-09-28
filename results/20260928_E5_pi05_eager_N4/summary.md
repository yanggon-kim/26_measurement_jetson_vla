# 20260928_E5_pi05_eager_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-28T20:00:41Z, power mode NV Power Mode: MAXN, model pi05, rung / mode none (eager, no ladder rung)
* **Command:** `bench/parse_nsys.py results/20260928_E5_pi05_eager_N4 --tag eager_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 24.672 | 185.1733 | 0.0757 | 185.1733 | 20 | 10 | 2259.9896 | 4234.0 |
| 52.064 | 236.6864 | 0.6495 | 236.6864 | 8 | 4 | 127.7522 | 1102.25 |
| 1.376 | 2.6355 | 0.011 | 2.6355 | 6 | 3 | 238.0464 | 1650.0 |
| 1.376 | 2.4709 | 0.0011 | 2.4709 | 2 | 1 | 2163.8787 | 879.0 |
|  |  | 0.0893 | 496.0709 | 1 | 1 | 5060.0466 | 6975.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-28: null rung replaced by its meaning (writer fixed in bench/parse_nsys.py / nsys_target.py)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
