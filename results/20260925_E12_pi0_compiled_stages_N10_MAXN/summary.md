# 20260925_E12_pi0_compiled_stages_N10_MAXN

* **What:** T014 Step 4 -- compiled stages alone (E12b harness, graphs_both)
* **Conditions:** date 2026-09-26T06:02:19Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12_capped_compiled_stages.py --frames 60 --outdir results/20260925_E12_pi0_compiled_stages_N10_MAXN`

Headline — first rows of `stages.csv` (3 rows):

| condition | stage | n | median_ms | p5_ms | p95_ms | mean_ms | period_ms |
|---|---|---|---|---|---|---|---|
| prefix_alone | prefix | 60 | 246.788 | 246.721 | 247.1327 | 246.8255 | 246.8291 |
| expert_alone | expert | 60 | 101.4767 | 101.3723 | 101.716 | 101.5185 | 101.523 |
| serial | frame | 60 | 348.7324 | 348.6198 | 348.8878 | 348.7426 | 348.8374 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
