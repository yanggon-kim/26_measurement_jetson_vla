# 20260925_E12_pi0_compiled_stages_N10_MAXN_green8_8

* **What:** T014 Step 4 -- compiled stages alone (E12b harness, graphs_both)
* **Conditions:** date 2026-09-26T06:07:53Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12_capped_compiled_stages.py --frames 60 --split 8/8 --outdir results/20260925_E12_pi0_compiled_stages_N10_MAXN_green8_8`

Headline — first rows of `stages.csv` (3 rows):

| condition | stage | n | median_ms | p5_ms | p95_ms | mean_ms | period_ms |
|---|---|---|---|---|---|---|---|
| prefix_alone | prefix | 60 | 327.6974 | 327.5934 | 329.1008 | 327.8356 | 327.8366 |
| expert_alone | expert | 60 | 149.4053 | 149.3024 | 149.7123 | 149.4714 | 149.4765 |
| serial | frame | 60 | 478.609 | 478.4528 | 478.8717 | 478.6116 | 478.7363 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
