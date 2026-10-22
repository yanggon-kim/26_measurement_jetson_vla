# E3 groot_n15 eager N sweep (T004) — eager_fp32, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 18.99 | 18.85–19.13 | 13.4 % |
| vlm_prefix | 31.06 | 30.94–31.21 | 21.9 % |
| ae_total | 52.59 | 52.41–52.80 | 37.1 % |
| other (e2e − named regions) | 39.13 | — | 27.6 % |
| e2e (model call) | 141.78 | — | 100.0 % |
| model_get_action | 124.37 | 124.02–124.76 | |
| outside_model | 17.38 | 9.04–18.74 | |
| vlm_backbone | 70.32 | 70.07–70.62 | |
| backbone_lm_head | 19.88 | 19.58–20.18 | |

ms per denoising step: mean of the per-step medians 11.18 (step 0 11.16, last 11.13); slope over N 11.2016 (r² 0.999999, ae_total = a + b·N, a = 7.7684 ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.121 | compute (torch.bfloat16) | 18.99 | 3.71 |
| vlm_prefix | 7.827 | compute (torch.bfloat16) | 31.06 | 3.97 |
| ae_step | 4.352 | memory (torch.bfloat16) | 11.20 | 2.57 |
| ae_total | 6.556 | compute (torch.bfloat16) | 52.59 | 8.02 |
| ae_total_restream | 17.410 | memory (torch.bfloat16) | 44.73 | 2.57 |

Tokens: {"prefix_tokens_P": 814, "P_dit_encoder": 814, "suffix_tokens_S": 49}.
Throttled per N: {8: False, 10: False, 6: False, 1: False, 2: False, 4: False}. Memory-path state per N: {8: 'normal->normal', 10: 'normal->normal', 6: 'normal->normal', 1: 'normal->normal', 2: 'normal->normal', 4: 'normal->normal'}.
Peak memory at N = 4: 5.93 GB allocated; control frequency 7.07 Hz (timed loop).
