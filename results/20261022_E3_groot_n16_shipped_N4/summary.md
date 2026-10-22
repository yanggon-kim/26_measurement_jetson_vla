# E3 groot_n16 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 21.06 | 20.86–21.60 | 15.5 % |
| vlm_prefix | 24.45 | 24.25–24.90 | 18.0 % |
| ae_total | 74.75 | 74.52–75.04 | 54.9 % |
| other (e2e − named regions) | 15.89 | — | 11.7 % |
| e2e (model call) | 136.14 | — | 100.0 % |
| model_get_action | 121.69 | 121.24–122.98 | |
| outside_model | 14.33 | 13.71–15.35 | |
| vlm_backbone | 45.83 | 45.50–46.82 | |

ms per denoising step: instrumented median 18.64 (step 0 18.64, last 18.61); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 6.520 | compute (torch.bfloat16) | 21.06 | 3.23 |
| vlm_prefix | 8.644 | memory (torch.bfloat16) | 24.45 | 2.83 |
| ae_step | 10.557 | memory | — | — |
| ae_total | 10.982 | memory (torch.bfloat16) | 74.75 | 6.81 |
| ae_total_restream | 42.229 | memory (torch.bfloat16) | 74.75 | 1.77 |

Tokens: {"prefix_tokens_P": 284, "P_dit_encoder": 284, "suffix_tokens_S": 51}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 6.96 GB allocated; control frequency 7.33 Hz (timed loop).
