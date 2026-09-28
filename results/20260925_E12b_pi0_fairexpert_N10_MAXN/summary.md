# 20260925_E12b_pi0_fairexpert_N10_MAXN

* **What:** E12b M1 -- the fair GPU expert (T015)
* **Conditions:** date 2026-09-25T17:39:25Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12b_m1.py --outdir results/20260925_E12b_pi0_fairexpert_N10_MAXN`

Headline — first rows of `equivalence.csv` (10 rows):

| variant | path | check | split | frame | obs_id | seed | max_abs_diff |
|---|---|---|---|---|---|---|---|
| bf16_both+hoist | compiled | hoist | full_gpu | 0 | obs_0 | 0 | 0.0 |
| bf16_both+stock | compiled | stock_run_to_run | full_gpu | 0 | obs_0 | 0 | 0.0 |
| bf16_both+hoist | compiled | hoist | full_gpu | 1 | obs_1 | 1 | 0.0 |
| bf16_both+stock | compiled | stock_run_to_run | full_gpu | 1 | obs_1 | 1 | 0.0 |
| bf16_both+hoist | compiled | hoist | full_gpu | 2 | obs_2 | 2 | 0.0 |
| bf16_both+stock | compiled | stock_run_to_run | full_gpu | 2 | obs_2 | 2 | 0.0 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; deviations: ["A tegrastats process was started by hand for under one second during the `pre stock prefix` loop (the operator checking the fastest interval). E7 measured tegrastats' cost at ~0.19 %; that loop's me

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
