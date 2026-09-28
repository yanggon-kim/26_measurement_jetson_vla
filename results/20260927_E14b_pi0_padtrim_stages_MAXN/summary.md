# 20260927_E14b_pi0_padtrim_stages_MAXN

* **What:** T017 step 2a/2c: stages per variant, bracketed
* **Conditions:** date 2026-09-27T23:29:27Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e14b_padtrim.py --mode time --frames 60 --outdir results/20260927_E14b_pi0_padtrim_stages_MAXN`

Headline — first rows of `stages.csv` (24 rows):

| variant | stage | n | median_ms | p5_ms | p95_ms | mean_ms |
|---|---|---|---|---|---|---|
| pad | vision | 180 | 54.8099 | 54.7717 | 55.0128 | 54.8398 |
| pad | vlm_prefix | 180 | 190.2236 | 189.8341 | 192.9426 | 190.6217 |
| pad | split_sum | 180 | 245.0459 | 244.6318 | 247.88 | 245.4615 |
| pad | prefix | 180 | 245.4129 | 244.995 | 246.8468 | 245.5687 |
| pad | expert | 180 | 101.5954 | 101.3543 | 102.1613 | 101.6533 |
| pad | serial | 180 | 347.637 | 347.0075 | 348.9111 | 347.7541 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; nocam_note: LeRobot's _preprocess_images builds the -1 dummy for the empty camera outside every timed region; it is removed from the list before the image encoder -- no SigLIP / projector call, no tokens, no mask; vision_bitwise_note: added after the stages run: bench/e14b_padtrim.py --mode visioncheck (row 9 sub-item), same day, MAXN

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
