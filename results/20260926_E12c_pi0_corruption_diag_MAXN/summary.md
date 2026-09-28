# 20260926_E12c_pi0_corruption_diag_MAXN

* **What:** T016 step 5 (EV-13): E12b compiled co-run corruption, 2x2 diagnosis
* **Conditions:** date 2026-09-27T02:08:17Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12c_diagnose.py --pool own --kv double --frames 100 --outdir results/20260926_E12c_pi0_corruption_diag_MAXN/own_pool_double_kv`
* **Sub-runs:** own_pool_double_kv, own_pool_single_kv, shared_pool_double_kv, shared_pool_single_kv

Headline — first rows of `own_pool_double_kv/frames.csv` (100 rows):

| frame | pipeline_bitwise_equal_… | pipeline_max_abs_diff | pipeline_nonfinite | stale_source_frame | stale_differs |
|---|---|---|---|---|---|
| 0 | True | 0.0 | 0 | 0 | False |
| 1 | True | 0.0 | 0 | 0 | True |
| 2 | True | 0.0 | 0 | 0 | True |
| 3 | True | 0.0 | 0 | 3 | False |
| 4 | True | 0.0 | 0 | 3 | True |
| 5 | True | 0.0 | 0 | 3 | True |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
