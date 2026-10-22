# E3 groot_n17 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 28.86 | 28.72–29.61 | 19.0 % |
| vlm_prefix | 25.06 | 24.90–26.27 | 16.5 % |
| ae_total | 72.81 | 72.60–76.65 | 48.0 % |
| other (e2e − named regions) | 24.87 | — | 16.4 % |
| e2e (model call) | 151.60 | — | 100.0 % |
| model_get_action | 134.04 | 133.70–139.89 | |
| outside_model | 17.48 | 17.11–18.25 | |
| vlm_backbone | 60.23 | 59.92–62.19 | |
| backbone_lm_head | 3.42 | 3.39–3.50 | |

ms per denoising step: mean of the per-step medians 17.33 (step 0 17.32, last 17.29); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.386 | compute (torch.bfloat16) | 28.86 | 5.36 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 25.06 | 4.02 |
| ae_step | 8.526 | memory (torch.bfloat16) | 17.33 | 2.03 |
| ae_total | 10.101 | memory (torch.bfloat16) | 72.81 | 7.21 |
| ae_total_restream | 34.103 | memory (torch.bfloat16) | 69.33 | 2.03 |

Tokens: {"prefix_tokens_P": 273, "P_dit_encoder": 273, "suffix_tokens_S": 41}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 6.62 GB allocated; control frequency 6.56 Hz (timed loop).
