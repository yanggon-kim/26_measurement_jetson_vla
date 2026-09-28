# 20260922_E4d_smolvla_paper_config_N10

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T05:37:34Z, power mode NV Power Mode: MAXN, model smolvla
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_smolvla_paper_config_N10 --model smolvla --steps 10 --tf32 on`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 8765.6 | 8765 | 8765.0 | 8768.0 | 83 |
| VDD_CPU_CV | 2851.4 | 2789 | 2789.0 | 3187.0 | 83 |
| VIN_SYS_5V0 | 5151.9 | 5141 | 5141.0 | 5242.0 | 83 |
| VDDQ_VDD2_1V8AO | 935.0 | 907 | 907.0 | 1008.0 | 83 |
| module | 17703.9 | 17602 | 17602.0 | 18205.0 | 83 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
