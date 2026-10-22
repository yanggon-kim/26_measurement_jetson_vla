# E3 smolvla eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 15.02 | 14.91–15.15 | 8.9 % |
| vlm_prefix | 14.09 | 13.91–14.36 | 8.4 % |
| ae_total | 135.34 | 134.55–136.61 | 80.6 % |
| other (e2e − named regions) | 3.39 | — | 2.0 % |
| e2e (model call) | 167.84 | — | 100.0 % |
| preprocess_wall | 0.49 | 0.46–0.52 | |
| postprocess_wall | 0.10 | 0.10–0.11 | |
| e2e_full_wall | 168.44 | 167.59–169.66 | |

ms per denoising step: instrumented median 13.44 (step 0 13.77, last 13.37); slope over N 13.3934 (r² 0.998375, ae_total = a + b·N, a = 2.4075 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 3.318 | compute (torch.bfloat16) | 15.02 | 4.53 |
| vlm_prefix | 1.218 | memory (torch.bfloat16) | 14.09 | 11.57 |
| ae_step | 0.812 | memory (torch.bfloat16) | 13.39 | 16.48 |
| ae_total | 0.849 | compute (torch.bfloat16) | 135.34 | 159.41 |
| ae_total_restream | 8.125 | memory (torch.bfloat16) | 135.34 | 16.66 |

Tokens: {"prefix_tokens_P": 177, "prefix_tokens_valid": 151, "suffix_tokens_S": 50, "suffix_tokens_valid": 50, "n_images": 2, "language_tokens": 48, "language_tokens_valid": 22, "chunk_size": 50}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 10: 1.12 GB allocated; control frequency 5.93 Hz (timed loop).
