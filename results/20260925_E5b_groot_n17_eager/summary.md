# 20260925_E5b_groot_n17_eager

* **What:** E5b (T009 step 3) -- instruction-level analysis, GR00T N1.7 action head
* **Conditions:** power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager

Headline — first rows of `stall_attribution.csv` (28 rows):

| model | mode | region | kernel | duration_us | launches | sm_throughput_pct | launch_bound_s1 |
|---|---|---|---|---|---|---|---|
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |
| groot_n17 | eager | ae_total | ampere_bf16_s16816gemm_… | 3881.19 | 8 | 65.57 |  |

**Anything odd:** raw_files_note: 2026-09-28 completeness audit: raw_files listing added (sha256 of every file in the directory)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
