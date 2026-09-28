# 20260922_E4d_pi0_paper_config_N10_fp32

* **What:** E4d (paper configuration reproduction) — task T004
* **Conditions:** date 2026-09-23T02:15:28Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/paper_repro.py --outdir results/20260922_E4d_pi0_paper_config_N10_fp32 --steps 10 --tf32 off`

Headline — first rows of `power.csv` (5 rows):

| rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|
| VDD_GPU_SOC | 15541.9 | 15544 | 15538.0 | 15544.0 | 57 |
| VDD_CPU_CV | 2446.5 | 2391 | 2390.0 | 2790.0 | 57 |
| VIN_SYS_5V0 | 6147.4 | 6149 | 6149.0 | 6149.0 | 57 |
| VDDQ_VDD2_1V8AO | 1512.0 | 1512 | 1512.0 | 1512.0 | 57 |
| module | 25647.8 | 25596 | 25589.0 | 25995.0 | 57 |

**Anything odd:** notes: ['Their energy column takes only three distinct values across seven models (1866/662/355 J), so it appears assigned per model-class rather than measured per model. Ours is four INA3221 rails sampled d; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
