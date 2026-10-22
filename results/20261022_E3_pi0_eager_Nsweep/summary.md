# E3 pi0 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 175.90 | 175.80–176.00 | 41.3 % |
| vlm_prefix | 120.43 | 119.96–120.90 | 28.3 % |
| ae_total | 126.62 | 124.44–143.23 | 29.7 % |
| other (e2e − named regions) | 2.98 | — | 0.7 % |
| e2e (model call) | 425.93 | — | 100.0 % |
| preprocess_wall | 0.61 | 0.58–0.67 | |
| postprocess_wall | 0.14 | 0.13–0.16 | |
| e2e_full_wall | 426.68 | 424.36–443.58 | |

ms per denoising step: instrumented median 12.61 (step 0 13.46, last 12.43); slope over N 12.5276 (r² 0.997957, ae_total = a + b·N, a = 1.6486 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 102.774 | compute (torch.float32) | 175.90 | 1.71 |
| vlm_prefix | 25.676 | compute (torch.bfloat16) | 120.43 | 4.69 |
| ae_step | 2.521 | memory (torch.bfloat16) | 12.53 | 4.97 |
| ae_total | 2.975 | compute (torch.bfloat16) | 126.62 | 42.57 |
| ae_total_restream | 25.212 | memory (torch.bfloat16) | 126.62 | 5.02 |

Tokens: {"prefix_tokens_P": 816, "prefix_tokens_valid": 533, "suffix_tokens_S": 51, "suffix_tokens_valid": 51, "n_images": 3, "language_tokens": 48, "language_tokens_valid": 21, "chunk_size": 50}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 10: 9.19 GB allocated; control frequency 2.33 Hz (timed loop).
