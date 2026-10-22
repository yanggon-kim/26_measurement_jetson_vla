# E3 groot_n15 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 19.11 | 19.00–19.27 | 13.3 % |
| vlm_prefix | 51.61 | 51.30–52.09 | 36.0 % |
| ae_total | 53.33 | 53.14–53.56 | 37.2 % |
| other (e2e − named regions) | 19.12 | — | 13.4 % |
| e2e (model call) | 143.17 | — | 100.0 % |
| model_get_action | 125.97 | 125.43–126.41 | |
| outside_model | 17.23 | 8.93–19.15 | |
| vlm_backbone | 71.12 | 70.78–71.50 | |

ms per denoising step: instrumented median 11.37 (step 0 11.36, last 11.33); slope over N 11.4102 (r² 0.999999, ae_total = a + b·N, a = 7.6999 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.121 | compute (torch.bfloat16) | 19.11 | 3.73 |
| vlm_prefix | 11.724 | compute (torch.bfloat16) | 51.61 | 4.40 |
| ae_step | 6.325 | memory (torch.bfloat16) | 11.41 | 1.80 |
| ae_total | 8.292 | memory (torch.bfloat16) | 53.33 | 6.43 |
| ae_total_restream | 25.298 | memory (torch.bfloat16) | 53.33 | 2.11 |

Tokens: {"prefix_tokens_P": 814, "P_dit_encoder": 814, "suffix_tokens_S": 49}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 5.93 GB allocated; control frequency 7.01 Hz (timed loop).
