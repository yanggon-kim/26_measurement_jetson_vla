# E11d - GR00T N1.7 prefix vs input (eager)

Run directory: `20260928_E11d_groot_n17_eager_prefixsweep`  
Written by `bench/groot_phases.py` at 2026-09-28T21:22:40Z.

## Answer

| n_images | prompt_label | prompt_tokens | image_tokens | P_processed | P_valid | region | median_ms | p5_ms | p95_ms | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | short | 14 | 256 | 270 | 270 | vision | 69.936 | 69.713 | 70.344 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | vlm_prefix | 157.912 | 157.243 | 160.246 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | ae_total | 210.507 | 209.022 | 214.152 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | ae_step | 48.672 | 48.301 | 49.417 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | e2e | 370.862 | 368.787 | 375.190 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | vision | 70.047 | 69.750 | 73.755 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | vlm_prefix | 158.080 | 157.270 | 162.721 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | ae_total | 210.430 | 209.195 | 217.534 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | ae_step | 48.630 | 48.346 | 50.193 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | e2e | 370.593 | 369.088 | 379.700 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | vision | 70.260 | 69.846 | 71.333 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | vlm_prefix | 155.247 | 154.299 | 157.784 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | ae_total | 216.652 | 214.927 | 219.932 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | ae_step | 50.308 | 49.923 | 51.018 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | e2e | 374.124 | 371.759 | 381.358 | 30 |

## Fits

| fit | region | x | fit_a_ms | fit_b | unit_b | fit_r2 | n_points | note |
|---|---|---|---|---|---|---|---|---|
| vision_vs_images | vision | n_images |  |  |  |  | 1 | x takes 1 value(s) across the accepted points; no slope |
| vlm_prefix_vs_P | vlm_prefix | P_processed | 170.48577 | -0.04601645 | ms_per_token | 0.99072 | 3 |  |
| ae_step_vs_P | ae_step | P_processed | 41.11269 | 0.02777083 | ms_per_token | 0.995728 | 3 |  |
| ae_step_vs_P | ae_total | P_processed | 182.32475 | 0.10367783 | ms_per_token | 0.997043 | 3 |  |
| ae_step_vs_P | e2e | P_processed | 355.31189 | 0.05680083 | ms_per_token | 0.987482 | 3 |  |

## Conditions

rung `eager` (eager), N = 4, MAXN. The embodiment declares cameras ['exterior_image_1_left', 'wrist_image_left']; the 1-image point drops `wrist_image_left`, the 3-image point adds `extra_camera_0` (a copy of `exterior_image_1_left`). vlm_prefix = the whole backbone (vision INSIDE, T012 step 4); vision = the image-encoder calls within it. Prompt limit: tokenizer model_max_length = 262144 (policy.processor.processor.tokenizer).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung eager --tf32 on --e11 prefixsweep --warmup 5 --timed 30 --outdir results/20260928_E11d_groot_n17_eager_prefixsweep
```

## Not measured

- 1img_short: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[ 21, 228,  77],
- 1img_typical: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[ 21, 228,  77],
- 1img_long: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[ 21, 228,  77],
- 3img_short: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
- 3img_typical: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
- 3img_long: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
