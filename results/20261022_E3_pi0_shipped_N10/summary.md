# E3 pi0 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 175.11 | 175.00–175.22 | 40.7 % |
| vlm_prefix | 125.96 | 125.38–126.63 | 29.3 % |
| ae_total | 125.86 | 124.36–128.27 | 29.3 % |
| other (e2e − named regions) | 2.85 | — | 0.7 % |
| e2e (model call) | 429.78 | — | 100.0 % |
| preprocess_wall | 0.59 | 0.55–0.63 | |
| postprocess_wall | 0.13 | 0.12–0.14 | |
| e2e_full_wall | 430.48 | 428.78–433.27 | |

ms per denoising step: instrumented median 12.49 (step 0 13.59, last 12.28); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 102.774 | compute (torch.float32) | 175.11 | 1.70 |
| vlm_prefix | 25.676 | compute (torch.bfloat16) | 125.96 | 4.91 |
| ae_step | 2.521 | memory | — | — |
| ae_total | 2.975 | compute (torch.bfloat16) | 125.86 | 42.31 |
| ae_total_restream | 25.212 | memory (torch.bfloat16) | 125.86 | 4.99 |

Tokens: {"prefix_tokens_P": 816, "prefix_tokens_valid": 533, "suffix_tokens_S": 51, "suffix_tokens_valid": 51, "n_images": 3, "language_tokens": 48, "language_tokens_valid": 21, "chunk_size": 50}.
Throttled per N: {10: False}. Memory-path state per N: {10: 'normal->normal'}.
Peak memory at N = 10: 9.19 GB allocated; control frequency 2.32 Hz (timed loop).
