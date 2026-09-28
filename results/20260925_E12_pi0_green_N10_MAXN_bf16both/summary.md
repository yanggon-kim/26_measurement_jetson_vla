# 20260925_E12_pi0_green_N10_MAXN_bf16both

* **What:** E12 Step 3 -- SM partitioning via CUDA green contexts
* **Conditions:** power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun_green.py --rung bf16_both --steps 10 --sms 4,6,8,10,12,16 --seconds 20 --outdir results/20260925_E12_pi0_green_N10_MAXN_bf16both`

Headline — first rows of `sm_sensitivity.csv` (12 rows):

| stage | want_sm | granted_sm | rung | N | power_mode | n | median_ms |
|---|---|---|---|---|---|---|---|
| prefix | 4 | 4 | bf16_both | 10 | NV Power Mode: MAXN | 41 | 491.6007 |
| expert | 4 | 4 | bf16_both | 10 | NV Power Mode: MAXN | 171 | 117.0474 |
| prefix | 6 | 6 | bf16_both | 10 | NV Power Mode: MAXN | 53 | 378.0635 |
| expert | 6 | 6 | bf16_both | 10 | NV Power Mode: MAXN | 175 | 114.6919 |
| prefix | 8 | 8 | bf16_both | 10 | NV Power Mode: MAXN | 63 | 320.7083 |
| expert | 8 | 8 | bf16_both | 10 | NV Power Mode: MAXN | 176 | 113.7003 |

**Anything odd:** rung_note: bf16_both: prep=[vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)] compiled=['model.denoise_step@reduce-overhead', 'pg.embed_; not_measured: ["Two stages running CONCURRENTLY in separate partitions. This run establishes each stage's own SM demand, which is the necessary condition for contention; it does not exhibit interference. Running bo; caveat: sm_efficiency below ~0.3 for a stage means that stage cannot be the victim OR the cause of CTA-level contention on this device, because it does not use the SMs it already has. That is a statement abou

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
