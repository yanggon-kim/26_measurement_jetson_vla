# 20260922_E5_pi0_eager_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-24T08:37:34Z, model pi0
* **Command:** `bench/parse_nsys.py results/20260922_E5_pi0_eager_N10 --tag eager_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (5 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 59.136 | 656.8192 | 0.6494 | 656.8192 | 44 | 22 | 354.6581 | 9711.0 |
| 61.632 | 720.0318 | 0.8417 | 720.0318 | 20 | 10 | 135.3738 | 968.7 |
| 1.344 | 2.787 | 0.0115 | 2.787 | 6 | 3 | 240.0318 | 1650.0 |
| 1.344 | 3.8267 | 0.0162 | 3.8267 | 2 | 1 | 232.4097 | 939.0 |
|  |  | 0.5466 | 1466.4198 | 1 | 1 | 1216.2724 | 12318.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed; raw_note: 2026-09-28 completeness audit: this capture's raw stdout / stats logs were not kept (no raw/); its nsys CSV exports under nsys/ are the committed record; the .nsys-rep/.sqlite are never committed (pro

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
