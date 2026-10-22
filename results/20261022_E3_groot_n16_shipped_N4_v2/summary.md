# E3 groot_n16 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 21.20 | 21.09–21.40 | 15.7 % |
| vlm_prefix | 20.64 | 20.50–20.99 | 15.3 % |
| ae_total | 73.42 | 73.20–73.70 | 54.3 % |
| other (e2e − named regions) | 20.02 | — | 14.8 % |
| e2e (model call) | 135.28 | — | 100.0 % |
| model_get_action | 120.63 | 120.35–121.12 | |
| outside_model | 14.55 | 13.82–15.54 | |
| vlm_backbone | 46.11 | 45.92–46.61 | |
| backbone_lm_head | 3.88 | 3.82–4.03 | |

ms per denoising step: mean of the per-step medians 18.32 (step 0 18.29, last 18.30); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 6.520 | compute (torch.bfloat16) | 21.20 | 3.25 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 20.64 | 3.31 |
| ae_step | 8.526 | memory (torch.bfloat16) | 18.32 | 2.15 |
| ae_total | 8.541 | memory (torch.bfloat16) | 73.42 | 8.60 |
| ae_total_restream | 34.104 | memory (torch.bfloat16) | 73.26 | 2.15 |

Tokens: {"prefix_tokens_P": 284, "P_dit_encoder": 284, "suffix_tokens_S": 51}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 6.96 GB allocated; control frequency 7.38 Hz (timed loop).
