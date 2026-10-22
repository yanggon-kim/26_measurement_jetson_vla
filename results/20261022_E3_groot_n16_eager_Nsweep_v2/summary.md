# E3 groot_n16 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 21.16 | 21.06–21.36 | 15.7 % |
| vlm_prefix | 20.49 | 20.34–20.91 | 15.2 % |
| ae_total | 73.24 | 73.07–73.52 | 54.2 % |
| other (e2e − named regions) | 20.15 | — | 14.9 % |
| e2e (model call) | 135.04 | — | 100.0 % |
| model_get_action | 120.31 | 120.04–120.78 | |
| outside_model | 14.67 | 14.27–15.77 | |
| vlm_backbone | 45.98 | 45.76–46.48 | |
| backbone_lm_head | 3.92 | 3.88–4.05 | |

ms per denoising step: mean of the per-step medians 18.26 (step 0 18.23, last 18.23); slope over N 18.199 (r² 0.999994, ae_total = a + b·N, a = 0.2078 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 6.520 | compute (torch.bfloat16) | 21.16 | 3.25 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 20.49 | 3.28 |
| ae_step | 8.526 | memory (torch.bfloat16) | 18.20 | 2.13 |
| ae_total | 8.541 | memory (torch.bfloat16) | 73.24 | 8.58 |
| ae_total_restream | 34.104 | memory (torch.bfloat16) | 73.06 | 2.14 |

Tokens: {"prefix_tokens_P": 284, "P_dit_encoder": 284, "suffix_tokens_S": 51}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 6.96 GB allocated; control frequency 7.38 Hz (timed loop).
