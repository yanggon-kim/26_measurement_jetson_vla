# 20260927_E12_pi0_compiled_stages_N10_MODE_15W

* **What:** T014 Step 4 -- compiled stages alone (E12b harness, graphs_both)
* **Conditions:** date 2026-09-28T03:59:40Z, power mode NV Power Mode: MODE_15W, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12_capped_compiled_stages.py --frames 60 --outdir results/20260927_E12_pi0_compiled_stages_N10_MODE_15W`

Headline — first rows of `stages.csv` (3 rows):

| condition | stage | n | median_ms | p5_ms | p95_ms | mean_ms | period_ms |
|---|---|---|---|---|---|---|---|
| prefix_alone | prefix | 60 | 1497.2183 | 1497.0331 | 1497.4199 | 1497.2243 | 1497.1967 |
| expert_alone | expert | 60 | 542.9468 | 542.7795 | 543.1479 | 542.9509 | 542.9506 |
| serial | frame | 60 | 2040.6521 | 2040.399 | 2040.8962 | 2040.6401 | 2040.932 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
