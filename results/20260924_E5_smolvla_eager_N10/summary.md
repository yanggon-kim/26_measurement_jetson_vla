# 20260924_E5_smolvla_eager_N10

* **What:** E5 (nsys timeline) — task T004/T008/T009
* **Conditions:** date 2026-09-25T02:42:02Z, power mode NV Power Mode: MAXN, model smolvla
* **Command:** `bench/parse_nsys.py results/20260924_E5_smolvla_eager_N10 --tag eager_N10 --steps 10 --inferences 2`

Headline — first rows of `timeline_summary.csv` (7 rows):

| gap_median_us | gap_total_ms | gpu_idle_frac | gpu_idle_ms | instances | instances_per_inference | kernel_union_ms | kernels_per_unit |
|---|---|---|---|---|---|---|---|
| 17.44 | 9.4106 | 0.3886 | 9.4106 | 8 | 4 | 14.8057 | 240.0 |
| 64.16 | 869.9842 | 0.923 | 869.9842 | 44 | 22 | 72.6255 | 11987.0 |
| 17.184 | 9.3876 | 0.388 | 9.3876 | 8 | 4 | 14.8071 | 241.0 |
| 64.48 | 865.2457 | 0.9273 | 865.2457 | 20 | 10 | 67.8172 | 1183.7 |
| 15.36 | 17.4018 | 0.3712 | 17.4018 | 4 | 2 | 29.4802 | 464.0 |
| 62.016 | 86.7521 | 0.8846 | 86.7521 | 2 | 1 | 11.3216 | 1287.0 |

**Anything odd:** notes: ['The .nsys-rep and .sqlite are NOT committed on this device: they embed the system-wide process environment including a session token, verified to survive `env -i`. They stay in scratch/ and are list; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
