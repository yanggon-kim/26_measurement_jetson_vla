# 20260922_E4d_smolvla_paper_config_N10_fp32

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T05:40:21Z, power mode NV Power Mode: MAXN, model smolvla
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_smolvla_paper_config_N10_fp32 --model smolvla --steps 10 --tf32 off`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 8802.5 | 8765 | 8765.0 | 9163.0 | 85 |
| VDD_CPU_CV | 2793.6 | 2789 | 2788.0 | 2789.0 | 85 |
| VIN_SYS_5V0 | 5139.8 | 5141 | 5141.0 | 5141.0 | 85 |
| VDDQ_VDD2_1V8AO | 909.4 | 907 | 907.0 | 907.0 | 85 |
| module | 17645.2 | 17602 | 17601.0 | 18000.0 | 85 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
