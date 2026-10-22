# 20261022_E4i_groot_n16_best_search — T005, groot_n16, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs` | measured | 125.95 (125.10–127.49) | 21.23 (21.10–21.56) | 20.64 (20.47–20.99) | 63.91 (63.62–64.26) | 20.08 (19.14–21.06) | 111.35 (110.99–112.14) | 57.4 | — | 98.31 | 0 | False | normal->normal |
| `ae_graphs_autotune` | measured | 112.44 (108.25–113.94) | 21.13 (21.03–21.59) | 20.64 (20.47–21.03) | 50.95 (50.78–51.25) | 19.70 (15.41–20.41) | 98.38 (98.10–99.30) | 51.8 | — | 169.26 | 0 | False | normal->normal |
| `ae_graphs_mpad` | measured | 127.32 (122.52–130.87) | 23.80 (23.52–24.18) | 21.35 (21.18–21.71) | 62.68 (62.47–64.80) | 19.44 (14.94–20.32) | 113.01 (112.47–116.13) | 55.5 | — | 106.44 | 0 | False | normal->normal |
| `ae_graphs_mpad_autotune` | measured | 119.46 (114.81–121.16) | 23.71 (23.54–24.25) | 21.17 (20.97–21.50) | 54.94 (54.69–55.15) | 19.53 (15.23–20.98) | 104.94 (104.47–105.88) | 52.3 | — | 181.86 | 0 | False | normal->normal |
| `all_compile_autotune_mpad` | measured | 111.58 (111.13–113.35) | 20.09 (19.88–20.32) | 20.18 (20.06–20.33) | 52.11 (52.01–52.32) | 19.21 (18.95–20.25) | 97.42 (97.13–97.83) | 53.5 | — | 264.75 | 0 | False | normal->normal |
| `llm_compile_autotune` | measured | 106.42 (101.99–107.70) | 21.23 (21.11–21.45) | 16.44 (16.33–16.71) | 49.47 (49.32–49.68) | 19.36 (14.91–20.06) | 92.76 (92.42–93.32) | 53.3 | — | 202.76 | 0 | False | normal->normal |
| `llm_graphs_autotune` | measured | 109.84 (105.48–111.66) | 21.27 (21.11–21.80) | 16.30 (16.20–16.50) | 52.19 (51.95–53.12) | 20.02 (15.80–20.70) | 95.46 (95.09–97.55) | 54.7 | — | 182.9 | 0 | False | normal->normal |
| `llm_graphs_mpad_autotune` | measured | 112.57 (108.12–114.07) | 23.91 (23.70–24.45) | 15.96 (15.82–16.26) | 53.53 (53.41–53.75) | 19.13 (14.81–20.10) | 98.49 (98.06–99.26) | 54.3 | — | 194.33 | 0 | False | normal->normal |
| `vlm_graphs_autotune` | measured | 107.67 (106.84–109.30) | 18.56 (18.47–18.72) | 17.76 (17.66–17.88) | 51.59 (51.39–51.86) | 19.75 (19.07–21.17) | 93.53 (93.34–93.89) | 55.2 | — | 249.24 | 0 | False | normal->normal |
| `vlm_graphs_mpad_autotune` | measured | 110.30 (105.88–111.12) | 22.95 (22.82–23.09) | 15.98 (15.86–16.18) | 52.13 (52.00–52.28) | 19.23 (14.85–19.97) | 96.17 (95.86–96.46) | 54.2 | — | 241.29 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_graphs_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.3438e-02 | 1.4745e-02 | True |
| `ae_graphs_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.3438e-02 | 1.4745e-02 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.4718e-02 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.4718e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.3438e-02 | 1.4406e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.3438e-02 | 1.4406e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.4711e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.4711e-02 | True |
| `all_compile_autotune_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.6764e-02 | True |
| `all_compile_autotune_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.6764e-02 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.6288e-02 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.6288e-02 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 3.1250e-02 | 1.7505e-02 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 3.1250e-02 | 1.7505e-02 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 1.9531e-02 | 1.3256e-02 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 1.9531e-02 | 1.3256e-02 | True |
| `vlm_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 4.1016e-02 | 1.7192e-02 | True |
| `vlm_graphs_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 4.1016e-02 | 1.7192e-02 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 3.5156e-02 | 1.9006e-02 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 3.5156e-02 | 1.9006e-02 | True |

## Preparations

- `ae_graphs_mpad`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.0445, 'ms_at_pad': 0.035}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.1239, 'ms_at_pad': 0.0782}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0615, 'ms_at_pad': 0.051}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 323, 'ms_at_M': 0.037, 'ms_at_pad': 0.0329}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.048, 'ms_at_pad': 0.0329}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.1056, 'ms_at_pad': 0.0615}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 4.152, 'ms_at_pad': 3.4339}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.04, 'ms_at_pad': 0.0371}}
- `ae_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.0445, 'ms_at_pad': 0.0352}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.1241, 'ms_at_pad': 0.079}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.062, 'ms_at_pad': 0.0504}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0366, 'ms_at_pad': 0.0323}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.0481, 'ms_at_pad': 0.0323}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.106, 'ms_at_pad': 0.0609}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 3.9652, 'ms_at_pad': 3.4366}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.0404, 'ms_at_pad': 0.0383}}
- `all_compile_autotune_mpad`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.0464, 'ms_at_pad': 0.0356}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.1242, 'ms_at_pad': 0.0789}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 323, 'ms_at_M': 0.0608, 'ms_at_pad': 0.0509}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 323, 'ms_at_M': 0.0371, 'ms_at_pad': 0.0329}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.048, 'ms_at_pad': 0.0328}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 332, 'ms_at_M': 0.1062, 'ms_at_pad': 0.0618}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 4.0303, 'ms_at_pad': 3.4186}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.0402, 'ms_at_pad': 0.0377}}
- `llm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.0455, 'ms_at_pad': 0.0355}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.1258, 'ms_at_pad': 0.08}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0633, 'ms_at_pad': 0.0512}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0372, 'ms_at_pad': 0.033}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.0484, 'ms_at_pad': 0.0331}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 332, 'ms_at_M': 0.1064, 'ms_at_pad': 0.0621}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 4.0931, 'ms_at_pad': 3.4517}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.0414, 'ms_at_pad': 0.0376}}
- `vlm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.045, 'ms_at_pad': 0.0352}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.1234, 'ms_at_pad': 0.0783}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0622, 'ms_at_pad': 0.0514}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0366, 'ms_at_pad': 0.033}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.0484, 'ms_at_pad': 0.0331}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.1056, 'ms_at_pad': 0.0619}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 4.1302, 'ms_at_pad': 3.4444}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.041, 'ms_at_pad': 0.0371}}

