# 20260925_E12b_pi0_green_corun_N10_MAXN

* **What:** E12b M3 -- co-run inside green-context partitions + hog control (T015)
* **Conditions:** date 2026-09-25T18:49:20Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12b_m3.py --outdir results/20260925_E12b_pi0_green_corun_N10_MAXN --splits 12/4,10/6,8/8,6/10,14/2`

Headline — first rows of `corun_ratios.csv` (16 rows):

| mechanism | variant | split | prefix_sms | expert_sms | N | rung | valid |
|---|---|---|---|---|---|---|---|
| compiled_graphs_both | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | 10 | bf16_both | True |
| green_contexts_corun | bf16_both+hoist+owngrap… | 12/4 | 12 | 4 | 10 | bf16_both | True |
| bandwidth_hog_control | bf16_both+hoist+owngrap… | 12/4 | 12 | 4 | 10 | bf16_both | True |
| bandwidth_hog_control | bf16_both+hoist+owngrap… | 12/4 | 12 | 4 | 10 | bf16_both | True |
| green_contexts_corun | bf16_both+hoist+owngrap… | 10/6 | 10 | 6 | 10 | bf16_both | True |
| bandwidth_hog_control | bf16_both+hoist+owngrap… | 10/6 | 10 | 6 | 10 | bf16_both | True |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
