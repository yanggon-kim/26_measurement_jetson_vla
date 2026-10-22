# E3 groot_n17 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 28.93 | 28.76–29.86 | 18.6 % |
| vlm_prefix | 25.17 | 24.98–26.56 | 16.2 % |
| ae_total | 74.25 | 73.99–77.81 | 47.8 % |
| other (e2e − named regions) | 26.99 | — | 17.4 % |
| e2e (model call) | 155.32 | — | 100.0 % |
| model_get_action | 135.77 | 135.34–141.63 | |
| outside_model | 19.30 | 18.66–23.84 | |
| vlm_backbone | 60.44 | 60.13–63.12 | |
| backbone_lm_head | 3.46 | 3.42–3.60 | |

ms per denoising step: mean of the per-step medians 17.71 (step 0 17.66, last 17.68); slope over N 17.6979 (r² 0.999987, ae_total = a + b·N, a = 3.5202 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.386 | compute (torch.bfloat16) | 28.93 | 5.37 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 25.17 | 4.03 |
| ae_step | 8.526 | memory (torch.bfloat16) | 17.70 | 2.08 |
| ae_total | 10.101 | memory (torch.bfloat16) | 74.25 | 7.35 |
| ae_total_restream | 34.103 | memory (torch.bfloat16) | 70.82 | 2.08 |

Tokens: {"prefix_tokens_P": 273, "P_dit_encoder": 273, "suffix_tokens_S": 41}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 6.62 GB allocated; control frequency 6.39 Hz (timed loop).
