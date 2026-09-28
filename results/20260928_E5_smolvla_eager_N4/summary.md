# 20260928_E5_smolvla_eager_N4

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-28T20:05:57Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode none (eager, no ladder rung)
* **Command:** `bench/parse_nsys.py results/20260928_E5_smolvla_eager_N4 --tag eager_N4 --steps 4 --inferences 2`

Headline — first rows of `timeline_summary.csv` (7 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 19.616 | 10.0167 | 0.4037 | 10.0167 | 8 | 4 | 14.7962 | 240.0 |
| 65.024 | 356.1978 | 0.9178 | 356.1978 | 20 | 10 | 31.918 | 4883.0 |
| 19.616 | 10.0181 | 0.4037 | 10.0181 | 8 | 4 | 14.7977 | 241.0 |
| 65.92 | 352.6192 | 0.9285 | 352.6192 | 8 | 4 | 27.1593 | 1183.25 |
| 18.24 | 18.568 | 0.3863 | 18.568 | 4 | 2 | 29.494 | 464.0 |
| 63.072 | 88.0827 | 0.8866 | 88.0827 | 2 | 1 | 11.2689 | 1287.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-28: null rung replaced by its meaning (writer fixed in bench/parse_nsys.py / nsys_target.py)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
