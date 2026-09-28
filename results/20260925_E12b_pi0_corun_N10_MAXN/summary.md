# 20260925_E12b_pi0_corun_N10_MAXN

* **What:** E12b M2 -- compiled co-run, whole GPU (T015)
* **Conditions:** date 2026-09-25T17:56:46Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12b_m2.py --outdir results/20260925_E12b_pi0_corun_N10_MAXN --m1 results/20260925_E12b_pi0_fairexpert_N10_MAXN/summary.json`

Headline — first rows of `corun_ratios.csv` (6 rows):

| mechanism | variant | split | prefix_sms | expert_sms | power_mode | sm_count | N |
|---|---|---|---|---|---|---|---|
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |
| compiled_owngraph_expert | bf16_both+hoist+owngrap… | full_gpu | 16 | 16 | NV Power Mode: MAXN | 16 | 10 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
