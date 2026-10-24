# 20261024_E4i_groot_n15_rerun — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at_mpad` | measured | 105.44 (97.74–106.25) | 18.56 (18.43–18.69) | 26.57 (26.48–26.64) | 36.75 (36.68–36.85) | 23.56 (15.86–24.47) | 89.09 (88.80–89.29) | 41.2 | — | 156.52 | 0 | False | normal->normal |
| `ae_graphs_mpad` | measured | 115.56 (107.64–116.85) | 18.77 (18.65–18.90) | 30.41 (30.27–30.52) | 42.85 (42.69–42.99) | 23.50 (15.71–24.79) | 99.10 (98.77–99.42) | 43.2 | — | 59.52 | 0 | False | normal->normal |
| `ae_graphs_mpad_autotune` | measured | 111.93 (104.42–113.08) | 18.44 (18.33–18.59) | 30.13 (30.00–30.25) | 39.84 (39.75–39.95) | 23.50 (15.83–24.40) | 95.56 (95.23–95.81) | 41.7 | — | 111.37 | 0 | True | normal->normal |
| `llm_graphs_ae_at_mpad` | measured | 102.80 (101.53–104.02) | 18.43 (18.36–18.60) | 22.39 (22.25–22.57) | 37.28 (37.20–37.39) | 24.68 (23.42–25.78) | 85.38 (85.19–85.87) | 43.7 | — | 163.32 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.7515e-02 | 1.7535e-02 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.7515e-02 | 1.7535e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.8332e-02 | 1.6854e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.8332e-02 | 1.6854e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 2.0979e-02 | 1.6858e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 2.0979e-02 | 1.6858e-02 | True |
| `llm_graphs_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.5980e-02 | 1.7616e-02 | True |
| `llm_graphs_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.5980e-02 | 1.7616e-02 | True |

## Preparations

- `ae_at_mpad`: linear_mpad(backbone+action_head): 57 Linear modules padded; shapes padded: {'M=814 K=2048 N=6144 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.2099, 'ms_at_pad': 0.1975}, 'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 16.8761, 'ms_at_pad': 5.5219}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': 0.1358, 'ms_at_pad': 0.1224}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.3449, 'ms_at_pad': 0.3005}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0408, 'ms_at_pad': 0.0381}}
- `ae_graphs_mpad`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 16.9114, 'ms_at_pad': 5.7148}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 974, 'ms_at_M': 0.1359, 'ms_at_pad': 0.1208}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 0.3425, 'ms_at_pad': 0.2884}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0407, 'ms_at_pad': 0.0378}}
- `ae_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 17.4455, 'ms_at_pad': 5.695}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 926, 'ms_at_M': 0.1358, 'ms_at_pad': 0.1219}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1022, 'ms_at_M': 0.3235, 'ms_at_pad': 0.2857}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0407, 'ms_at_pad': 0.0383}}
- `llm_graphs_ae_at_mpad`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 16.6478, 'ms_at_pad': 5.5281}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': 0.1358, 'ms_at_pad': 0.1214}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.3385, 'ms_at_pad': 0.3151}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0408, 'ms_at_pad': 0.0383}}

