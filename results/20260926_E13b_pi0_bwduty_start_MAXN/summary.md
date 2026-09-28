# 20260926_E13b_pi0_bwduty_start_MAXN

* **What:** T016 step 1b (EV-02): prefix under SM-free bandwidth steal
* **Conditions:** date 2026-09-26T22:24:25Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e13_bwsteal.py --outdir results/20260926_E13b_pi0_bwduty_start_MAXN --mode duty --window start --win-ms 93 --load both --rates=-1,40 --frames 30 --harness-cores 0-3 --load-cores 4`

Headline — first rows of `sigma.csv` (2 rows):

| point | mode | load | target_GBps | window | load_alone_GBps_own | load_alone_GBps_emc | load_during_GBps_own |
|---|---|---|---|---|---|---|---|
| 0 | duty | both | max | start 93 ms | 46.75 | 50.25 | 27.37 |
| 1 | duty | both | 40.0 | start 93 ms | 14.97 | 31.46 | 14.3 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
