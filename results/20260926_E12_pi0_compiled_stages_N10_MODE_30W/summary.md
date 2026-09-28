# 20260926_E12_pi0_compiled_stages_N10_MODE_30W

* **What:** T014 Step 4 -- compiled stages alone (E12b harness, graphs_both)
* **Conditions:** date 2026-09-26T19:47:16Z, power mode NV Power Mode: MODE_30W, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12_capped_compiled_stages.py --frames 60 --outdir results/20260926_E12_pi0_compiled_stages_N10_MODE_30W`

Headline — first rows of `stages.csv` (3 rows):

| condition | stage | n | median_ms | p5_ms | p95_ms | mean_ms | period_ms |
|---|---|---|---|---|---|---|---|
| prefix_alone | prefix | 60 | 876.8417 | 876.7279 | 876.9506 | 876.8405 | 876.7904 |
| expert_alone | expert | 60 | 286.1256 | 286.014 | 286.2281 | 286.1256 | 286.1144 |
| serial | frame | 60 | 1163.2569 | 1163.0886 | 1163.4266 | 1163.2679 | 1163.4765 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
