# E3 pi05 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 174.69 | 174.58–174.81 | 12.1 % |
| vlm_prefix | 854.15 | 853.77–854.46 | 59.4 % |
| ae_total | 406.60 | 405.61–407.97 | 28.3 % |
| other (e2e − named regions) | 2.96 | — | 0.2 % |
| e2e (model call) | 1438.39 | — | 100.0 % |
| preprocess_wall | 0.94 | 0.91–1.01 | |
| postprocess_wall | 0.12 | 0.11–0.15 | |
| e2e_full_wall | 1439.47 | 1438.15–1441.03 | |

ms per denoising step: instrumented median 40.58 (step 0 40.93, last 40.51); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 102.774 | compute (torch.float32) | 174.69 | 1.70 |
| vlm_prefix | 618.477 | compute (torch.float32) | 854.15 | 1.38 |
| ae_step | 6.800 | memory | — | — |
| ae_total | 60.513 | compute (torch.float32) | 406.60 | 6.72 |
| ae_total_restream | 68.004 | memory (torch.float32) | 406.60 | 5.98 |

Tokens: {"prefix_tokens_P": 968, "prefix_tokens_valid": 574, "suffix_tokens_S": 50, "suffix_tokens_valid": 50, "n_images": 3, "language_tokens": 200, "language_tokens_valid": 62, "chunk_size": 50}.
Throttled per N: {10: False}. Memory-path state per N: {10: 'normal->normal'}.
Peak memory at N = 10: 17.05 GB allocated; control frequency 0.69 Hz (timed loop).
