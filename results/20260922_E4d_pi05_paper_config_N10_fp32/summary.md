# 20260922_E4d_pi05_paper_config_N10_fp32

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T03:09:12Z, power mode NV Power Mode: MAXN, model pi05
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_pi05_paper_config_N10_fp32 --model pi05 --steps 10 --tf32 off`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 24213.2 | 24293.0 | 23894.0 | 24293.0 | 20 |
| VDD_CPU_CV | 2409.1 | 2389.0 | 2389.0 | 2409.9 | 20 |
| VIN_SYS_5V0 | 7127.6 | 7168.0 | 7067.0 | 7168.0 | 20 |
| VDDQ_VDD2_1V8AO | 2197.8 | 2218.0 | 2117.0 | 2218.0 | 20 |
| module | 35947.7 | 36017.5 | 35467.0 | 36088.9 | 20 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
