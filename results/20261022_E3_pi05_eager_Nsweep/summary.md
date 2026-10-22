# E3 pi05 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 171.27 | 171.17–171.40 | 12.0 % |
| vlm_prefix | 854.52 | 854.08–854.95 | 59.6 % |
| ae_total | 404.48 | 403.33–408.12 | 28.2 % |
| other (e2e − named regions) | 2.87 | — | 0.2 % |
| e2e (model call) | 1433.14 | — | 100.0 % |
| preprocess_wall | 0.92 | 0.89–0.98 | |
| postprocess_wall | 0.12 | 0.12–0.13 | |
| e2e_full_wall | 1434.17 | 1432.84–1438.04 | |

ms per denoising step: instrumented median 40.37 (step 0 41.10, last 40.30); slope over N 40.3217 (r² 0.999989, ae_total = a + b·N, a = 1.4194 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 102.774 | compute (torch.float32) | 171.27 | 1.67 |
| vlm_prefix | 618.477 | compute (torch.float32) | 854.52 | 1.38 |
| ae_step | 6.800 | memory (torch.float32) | 40.32 | 5.93 |
| ae_total | 60.513 | compute (torch.float32) | 404.48 | 6.68 |
| ae_total_restream | 68.004 | memory (torch.float32) | 404.48 | 5.95 |

Tokens: {"prefix_tokens_P": 968, "prefix_tokens_valid": 574, "suffix_tokens_S": 50, "suffix_tokens_valid": 50, "n_images": 3, "language_tokens": 200, "language_tokens_valid": 62, "chunk_size": 50}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 10: 17.05 GB allocated; control frequency 0.70 Hz (timed loop).
