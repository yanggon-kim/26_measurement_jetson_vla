# E11d - GR00T N1.7 prefix vs input (vlm_compile)

Run directory: `20260928_E11d_groot_n17_vlm_compile_prefixsweep`  
Written by `bench/groot_phases.py` at 2026-09-28T21:28:40Z.

## Answer

| n_images | prompt_label | prompt_tokens | image_tokens | P_processed | P_valid | region | median_ms | p5_ms | p95_ms | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | short | 14 | 256 | 270 | 270 | vision | 70.966 | 70.756 | 71.281 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | vlm_prefix | 144.840 | 144.545 | 145.351 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | ae_total | 80.417 | 80.295 | 80.505 | 30 |
| 2 | short | 14 | 256 | 270 | 270 | e2e | 227.504 | 227.198 | 228.041 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | vision | 71.066 | 70.987 | 71.238 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | vlm_prefix | 152.126 | 151.676 | 152.566 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | ae_total | 80.539 | 80.395 | 80.622 | 30 |
| 2 | typical | 17 | 256 | 273 | 273 | e2e | 234.918 | 234.304 | 235.438 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | vision | 70.984 | 70.854 | 71.102 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | vlm_prefix | 148.028 | 147.695 | 148.423 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | ae_total | 85.094 | 84.813 | 85.246 | 30 |
| 2 | long | 75 | 256 | 331 | 331 | e2e | 235.325 | 234.877 | 235.692 | 30 |

## Fits

| fit | region | x | fit_a_ms | fit_b | unit_b | fit_r2 | n_points | note |
|---|---|---|---|---|---|---|---|---|
| vision_vs_images | vision | n_images |  |  |  |  | 1 | x takes 1 value(s) across the accepted points; no slope |
| vlm_prefix_vs_P | vlm_prefix | P_processed | 149.2124 | -0.00302376 | ms_per_token | 0.00081 | 3 |  |
| ae_step_vs_P | ae_step | P_processed |  |  |  |  | 0 | x takes 0 value(s) across the accepted points; no slope |
| ae_step_vs_P | ae_total | P_processed | 59.43432 | 0.07751379 | ms_per_token | 0.999567 | 3 |  |
| ae_step_vs_P | e2e | P_processed | 211.1073 | 0.07371367 | ms_per_token | 0.331474 | 3 |  |

## Conditions

rung `vlm_compile` (action_head.get_action@reduce-overhead+clone; backbone.model.language_model.forward@max-autotune-no-cudagraphs), N = 4, MAXN. The embodiment declares cameras ['exterior_image_1_left', 'wrist_image_left']; the 1-image point drops `wrist_image_left`, the 3-image point adds `extra_camera_0` (a copy of `exterior_image_1_left`). vlm_prefix = the whole backbone (vision INSIDE, T012 step 4); vision = the image-encoder calls within it. Prompt limit: tokenizer model_max_length = 262144 (policy.processor.processor.tokenizer).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs --e11 prefixsweep --warmup 5 --timed 30 --outdir results/20260928_E11d_groot_n17_vlm_compile_prefixsweep
```

## Not measured

- 1img_short: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[126, 179,  84],
- 1img_typical: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[126, 179,  84],
- 1img_long: refused by the checkpoint's processor -- AssertionError: wrist_image_left not in {'exterior_image_1_left': array([[[[126, 179,  84],
- 3img_short: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
- 3img_typical: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
- 3img_long: refused by the checkpoint's processor -- ValueError: extra image key(s) silently ignored by the processor: 256 image tokens with 3 images vs 256 with 2
- t_ae_step (compiled): the action head replays as one CUDA graph (reduce-overhead), so there is no per-step boundary to time; ae_total / N is a derivation, not a measurement, and is not written
