# E3 groot_n16 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 21.00 | 20.91–21.56 | 15.4 % |
| vlm_prefix | 24.89 | 24.72–25.32 | 18.3 % |
| ae_total | 74.62 | 74.39–74.91 | 54.8 % |
| other (e2e − named regions) | 15.72 | — | 11.5 % |
| e2e (model call) | 136.23 | — | 100.0 % |
| model_get_action | 122.00 | 121.65–123.16 | |
| outside_model | 14.22 | 13.96–14.79 | |
| vlm_backbone | 46.23 | 46.04–47.10 | |

ms per denoising step: instrumented median 18.61 (step 0 18.60, last 18.50); slope over N 18.5987 (r² 0.999996, ae_total = a + b·N, a = 0.1206 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 6.520 | compute (torch.bfloat16) | 21.00 | 3.22 |
| vlm_prefix | 8.644 | memory (torch.bfloat16) | 24.89 | 2.88 |
| ae_step | 10.557 | memory (torch.bfloat16) | 18.60 | 1.76 |
| ae_total | 10.982 | memory (torch.bfloat16) | 74.62 | 6.80 |
| ae_total_restream | 42.229 | memory (torch.bfloat16) | 74.62 | 1.77 |

Tokens: {"prefix_tokens_P": 284, "P_dit_encoder": 284, "suffix_tokens_S": 51}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 6.96 GB allocated; control frequency 7.32 Hz (timed loop).
