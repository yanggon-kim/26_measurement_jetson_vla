# 20260922_E4d_pi0_paper_config_N10

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T02:08:35Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_pi0_paper_config_N10 --steps 10 --tf32 on`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 14646.5 | 14741.0 | 14342.0 | 14747.0 | 66 |
| VDD_CPU_CV | 2668.2 | 2789.0 | 2390.0 | 2789.8 | 66 |
| VIN_SYS_5V0 | 6149.0 | 6149.0 | 6149.0 | 6149.0 | 66 |
| VDDQ_VDD2_1V8AO | 1519.7 | 1512.0 | 1512.0 | 1613.0 | 66 |
| module | 24983.3 | 25191.0 | 24399.0 | 25197.8 | 66 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
