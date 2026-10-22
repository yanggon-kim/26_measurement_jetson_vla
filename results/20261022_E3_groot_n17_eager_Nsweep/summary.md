# E3 groot_n17 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 29.26 | 29.01–29.78 | 17.6 % |
| vlm_prefix | 25.98 | 25.73–27.33 | 15.6 % |
| ae_total | 83.92 | 80.64–85.74 | 50.5 % |
| other (e2e − named regions) | 27.01 | — | 16.3 % |
| e2e (model call) | 166.17 | — | 100.0 % |
| model_get_action | 146.81 | 144.82–149.67 | |
| outside_model | 19.39 | 18.75–20.43 | |
| vlm_backbone | 61.85 | 61.28–63.72 | |

ms per denoising step: instrumented median 20.09 (step 0 18.43, last 20.43); slope over N 20.3496 (r² 0.999977, ae_total = a + b·N, a = 2.4754 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.386 | compute (torch.bfloat16) | 29.26 | 5.43 |
| vlm_prefix | 6.238 | memory (torch.bfloat16) | 25.98 | 4.17 |
| ae_step | 10.559 | memory (torch.bfloat16) | 20.35 | 1.93 |
| ae_total | 12.545 | memory (torch.bfloat16) | 83.92 | 6.69 |
| ae_total_restream | 42.238 | memory (torch.bfloat16) | 83.92 | 1.99 |

Tokens: {"prefix_tokens_P": 273, "P_dit_encoder": 273, "suffix_tokens_S": 41}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 6.62 GB allocated; control frequency 6.01 Hz (timed loop).
