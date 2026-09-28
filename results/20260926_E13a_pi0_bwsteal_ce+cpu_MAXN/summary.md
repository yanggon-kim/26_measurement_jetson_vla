# 20260926_E13a_pi0_bwsteal_ce+cpu_MAXN

* **What:** T016 step 1a (EV-02): prefix under SM-free bandwidth steal
* **Conditions:** date 2026-09-26T22:11:33Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e13_bwsteal.py --outdir results/20260926_E13a_pi0_bwsteal_ce+cpu_MAXN --mode sweep --load both --rates=0,20,40,60,80,100,120,140,-1 --frames 30 --harness-cores 0-3 --load-cores 4,`

Headline — first rows of `sigma.csv` (9 rows):

| point | mode | load | target_GBps | window | load_alone_GBps_own | load_alone_GBps_emc | load_during_GBps_own |
|---|---|---|---|---|---|---|---|
| 0 | sweep | both | 0.0 | always | 0.0 |  | 0.0 |
| 1 | sweep | both | 20.0 | always | 20.0 | 34.35 | 19.98 |
| 2 | sweep | both | 40.0 | always | 40.0 | 47.36 | 39.98 |
| 3 | sweep | both | 60.0 | always | 60.01 | 58.92 | 59.8 |
| 4 | sweep | both | 80.0 | always | 80.01 | 71.93 | 77.12 |
| 5 | sweep | both | 100.0 | always | 96.95 | 82.04 | 82.92 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
