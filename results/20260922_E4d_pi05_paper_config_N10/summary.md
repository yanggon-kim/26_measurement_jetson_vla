# 20260922_E4d_pi05_paper_config_N10

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T02:53:16Z, power mode NV Power Mode: MAXN, model pi05
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_pi05_paper_config_N10 --model pi05 --steps 10 --tf32 on`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 18294.5 | 18326 | 17928.0 | 18334.0 | 47 |
| VDD_CPU_CV | 2424.2 | 2390 | 2390.0 | 2789.0 | 47 |
| VIN_SYS_5V0 | 7240.9 | 7269 | 7157.0 | 7269.0 | 47 |
| VDDQ_VDD2_1V8AO | 2217.7 | 2218 | 2215.2 | 2218.0 | 47 |
| module | 30177.3 | 30203 | 30091.0 | 30212.0 | 47 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
