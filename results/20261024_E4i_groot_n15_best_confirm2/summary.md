# 20261024_E4i_groot_n15_best_confirm2 — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at_mpad` | measured | 92.50 (86.76–94.33) | 18.44 (18.32–18.60) | 26.50 (26.38–26.71) | 21.92 (21.83–22.03) | 25.55 (19.80–27.36) | 74.08 (73.75–74.59) | 29.6 | — | 69.47 | 0 | False | normal->normal |
| `llm_graphs_ae_at_mpad` | measured | 82.61 (78.28–88.35) | 18.39 (18.28–18.54) | 22.11 (21.93–22.29) | 21.57 (21.48–21.68) | 20.49 (16.27–26.22) | 69.17 (68.81–69.51) | 31.2 | — | 60.99 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at_mpad` | measured | 106.25 (98.47–107.84) | 18.55 (18.43–18.71) | 26.52 (26.40–26.62) | 36.84 (36.76–36.97) | 24.32 (16.50–25.52) | 89.16 (88.97–89.37) | 41.3 | — | 147.94 | 0 | False | normal->normal |
| `llm_graphs_ae_at_mpad` | measured | 100.34 (92.72–101.26) | 18.37 (18.28–18.49) | 22.27 (22.13–22.48) | 36.30 (36.22–36.41) | 23.39 (15.86–24.08) | 84.20 (83.93–84.52) | 43.1 | — | 157.42 | 0 | False | normal->normal |

Per-step time by slope: `ae_at_mpad`: 7.42 ms/step (ae_total, N = [2, 4, 6], intercept 7.11); `llm_graphs_ae_at_mpad`: 7.33 ms/step (ae_total, N = [2, 4, 6], intercept 6.94).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at_mpad` | measured | 121.93 (113.74–124.21) | 18.43 (18.29–18.61) | 26.52 (26.35–26.63) | 51.60 (51.46–51.73) | 25.30 (17.06–27.50) | 103.77 (103.50–104.10) | 49.7 | — | 158.94 | 0 | False | normal->normal |
| `llm_graphs_ae_at_mpad` | measured | 115.13 (106.95–116.97) | 18.34 (18.23–18.47) | 22.15 (21.99–22.36) | 50.87 (50.73–50.97) | 23.72 (15.60–25.57) | 98.48 (98.11–98.83) | 51.7 | — | 158.66 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.6231e-02 | 1.7894e-02 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.6231e-02 | 1.7894e-02 | True |
| `llm_graphs_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3652e-01 | True |
| `llm_graphs_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3652e-01 | True |

## Preparations

- `ae_at_mpad`: linear_mpad(backbone+action_head, replayed from the recorded best_bf16 table): 57 Linear modules padded; shapes padded: {'M=814 K=2048 N=6144 torch.bfloat16': {'pad_to': 974, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 974, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 974, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': nan, 'ms_at_pad': nan}}
- `llm_graphs_ae_at_mpad`: linear_mpad(backbone+action_head, replayed from the recorded best_bf16 table): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 974, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 974, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': nan, 'ms_at_pad': nan}}

