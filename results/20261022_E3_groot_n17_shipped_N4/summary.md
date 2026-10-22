# E3 groot_n17 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 28.82 | 28.65–30.29 | 19.0 % |
| vlm_prefix | 25.29 | 25.06–27.25 | 16.7 % |
| ae_total | 72.18 | 71.92–76.92 | 47.7 % |
| other (e2e − named regions) | 25.13 | — | 16.6 % |
| e2e (model call) | 151.42 | — | 100.0 % |
| model_get_action | 133.82 | 133.36–142.69 | |
| outside_model | 17.61 | 17.38–18.57 | |
| vlm_backbone | 60.60 | 60.23–63.94 | |

ms per denoising step: instrumented median 17.19 (step 0 17.21, last 17.12); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.386 | compute (torch.bfloat16) | 28.82 | 5.35 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 25.29 | 4.05 |
| ae_step | 10.559 | memory | — | — |
| ae_total | 12.545 | memory (torch.bfloat16) | 72.18 | 5.75 |
| ae_total_restream | 42.238 | memory (torch.bfloat16) | 72.18 | 1.71 |

Tokens: {"prefix_tokens_P": 273, "P_dit_encoder": 273, "suffix_tokens_S": 41}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 6.62 GB allocated; control frequency 6.53 Hz (timed loop).
