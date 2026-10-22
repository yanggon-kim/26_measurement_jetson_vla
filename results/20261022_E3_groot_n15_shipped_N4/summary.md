# E3 groot_n15 eager N sweep (T004) — shipped, MAXN, jetson_clocks, gate passed per N

At the model's default N = 4 (median of 100 timed iterations, CUDA events per region):

| region | median ms | p5–p95 ms | share of e2e |
|---|---|---|---|
| vision | 19.02 | 18.92–19.18 | 13.6 % |
| vlm_prefix | 52.05 | 51.78–52.41 | 37.3 % |
| ae_total | 51.10 | 50.94–51.33 | 36.6 % |
| other (e2e − named regions) | 17.47 | — | 12.5 % |
| e2e (model call) | 139.64 | — | 100.0 % |
| model_get_action | 123.94 | 123.62–124.43 | |
| outside_model | 15.73 | 8.22–16.44 | |
| vlm_backbone | 71.47 | 71.23–71.79 | |

ms per denoising step: instrumented median 10.85 (step 0 10.83, last 10.79); slope over N n/a (r² n/a, ae_total = a + b·N, a = n/a ms).

| part | floor ms (measured peaks) | bound | measured ms | gap ratio |
|---|---|---|---|---|
| vision | 5.121 | compute (torch.bfloat16) | 19.02 | 3.71 |
| vlm_prefix | 11.724 | compute (torch.bfloat16) | 52.05 | 4.44 |
| ae_step | 6.325 | memory | — | — |
| ae_total | 8.292 | memory (torch.bfloat16) | 51.10 | 6.16 |
| ae_total_restream | 25.298 | memory (torch.bfloat16) | 51.10 | 2.02 |

Tokens: {"prefix_tokens_P": 814, "P_dit_encoder": 814, "suffix_tokens_S": 49}.
Throttled per N: {4: False}. Memory-path state per N: {4: 'normal->normal'}.
Peak memory at N = 4: 5.93 GB allocated; control frequency 7.30 Hz (timed loop).
