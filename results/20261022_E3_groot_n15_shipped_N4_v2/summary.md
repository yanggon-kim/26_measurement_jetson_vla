# E3 groot_n15 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 18.88 | 18.75–19.00 | 13.4 % |
| vlm_prefix | 31.18 | 31.06–31.41 | 22.2 % |
| ae_total | 52.63 | 52.44–52.85 | 37.4 % |
| other (e2e − named regions) | 37.95 | — | 27.0 % |
| e2e (model call) | 140.65 | — | 100.0 % |
| model_get_action | 124.55 | 124.16–125.07 | |
| outside_model | 16.03 | 8.12–17.27 | |
| vlm_backbone | 70.44 | 70.25–70.84 | |
| backbone_lm_head | 19.97 | 19.76–20.29 | |

ms per denoising step: mean of the per-step medians 11.17 (step 0 11.16, last 11.13); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.121 | compute (torch.bfloat16) | 18.88 | 3.69 |
| vlm_prefix | 7.827 | compute (torch.bfloat16) | 31.18 | 3.98 |
| ae_step | 4.352 | memory (torch.bfloat16) | 11.17 | 2.57 |
| ae_total | 6.556 | compute (torch.bfloat16) | 52.63 | 8.03 |
| ae_total_restream | 17.410 | memory (torch.bfloat16) | 44.69 | 2.57 |

Tokens: {"prefix_tokens_P": 814, "P_dit_encoder": 814, "suffix_tokens_S": 49}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 5.93 GB allocated; control frequency 7.15 Hz (timed loop).
