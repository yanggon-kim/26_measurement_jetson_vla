# E3 smolvla eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 10 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 15.15 | 14.99–15.31 | 8.8 % |
| vlm_prefix | 14.31 | 13.98–14.89 | 8.3 % |
| ae_total | 138.84 | 136.81–144.07 | 80.8 % |
| other (e2e − named regions) | 3.54 | — | 2.1 % |
| e2e (model call) | 171.83 | — | 100.0 % |
| preprocess_wall | 0.51 | 0.48–0.56 | |
| postprocess_wall | 0.10 | 0.10–0.12 | |
| e2e_full_wall | 172.46 | 169.97–178.15 | |

ms per denoising step: instrumented median 13.83 (step 0 14.11, last 13.76); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 3.318 | compute (torch.bfloat16) | 15.15 | 4.56 |
| vlm_prefix | 1.218 | memory (torch.bfloat16) | 14.31 | 11.74 |
| ae_step | 0.812 | memory | — | — |
| ae_total | 0.849 | compute (torch.bfloat16) | 138.84 | 163.53 |
| ae_total_restream | 8.125 | memory (torch.bfloat16) | 138.84 | 17.09 |

Tokens: {"prefix_tokens_P": 177, "prefix_tokens_valid": 151, "suffix_tokens_S": 50, "suffix_tokens_valid": 50, "n_images": 2, "language_tokens": 48, "language_tokens_valid": 22, "chunk_size": 50}.
Throttled per N: {10: False}. Memory-path state per N: {10: 'normal->normal'}.
Peak memory at N = 10: 1.12 GB allocated; control frequency 5.75 Hz (timed loop).
