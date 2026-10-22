# 20261022_E4i_groot_n15_best_search — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs` | measured | 134.42 (126.42–135.94) | 19.08 (18.98–19.20) | 31.34 (31.20–31.54) | 45.09 (44.90–45.38) | 38.87 (30.73–40.07) | 117.81 (117.14–118.61) | 38.3 | — | 57.96 | 0 | False | normal->normal |
| `ae_graphs_autotune` | measured | 125.28 (117.13–126.59) | 18.77 (18.64–18.94) | 31.21 (31.09–31.46) | 36.41 (36.27–36.67) | 38.89 (30.65–40.13) | 108.34 (107.96–108.77) | 33.6 | — | 120.93 | 0 | False | normal->normal |
| `ae_graphs_mpad` | measured | 118.85 (110.54–120.53) | 19.11 (19.02–19.27) | 31.36 (31.24–31.50) | 44.83 (44.60–45.09) | 23.47 (15.35–24.96) | 102.49 (102.03–102.91) | 43.7 | — | 62.56 | 0 | True | normal->normal |
| `ae_graphs_mpad_autotune` | measured | 111.28 (106.39–113.20) | 19.08 (18.97–19.24) | 31.27 (31.13–31.53) | 37.63 (37.51–37.78) | 23.30 (18.07–24.53) | 95.16 (94.91–95.56) | 39.6 | — | 126.21 | 0 | True | normal->normal |
| `llm_compile_autotune` | measured | 117.23 (109.39–119.31) | 19.10 (19.00–19.25) | 24.43 (24.28–24.58) | 35.18 (35.07–35.34) | 38.51 (30.63–40.54) | 100.78 (100.42–101.17) | 34.9 | — | 158.81 | 0 | False | normal->normal |
| `llm_graphs_autotune` | measured | 116.56 (108.37–117.76) | 19.03 (18.90–19.19) | 24.08 (23.98–24.20) | 35.53 (35.39–35.67) | 37.91 (29.75–38.99) | 100.64 (100.19–101.23) | 35.3 | — | 141.79 | 0 | False | normal->normal |
| `llm_graphs_mpad_autotune` | measured | 100.80 (92.87–104.12) | 18.83 (18.72–19.01) | 18.80 (18.62–19.01) | 40.15 (40.03–42.64) | 23.03 (15.15–24.91) | 84.89 (84.44–87.78) | 47.3 | — | 144.85 | 0 | True | normal->normal |
| `vlm_graphs_autotune` | measured | 108.04 (99.45–109.32) | 12.50 (12.39–12.64) | 22.02 (21.90–22.17) | 35.47 (35.29–35.60) | 38.13 (29.83–39.27) | 92.18 (91.33–92.80) | 38.5 | — | 180.11 | 0 | False | normal->normal |
| `vlm_graphs_mpad_autotune` | measured | 99.18 (98.35–102.05) | 13.43 (13.31–13.60) | 24.14 (24.00–24.49) | 37.87 (37.76–38.23) | 23.78 (22.99–26.23) | 82.77 (82.43–83.60) | 45.8 | — | 191.86 | 0 | True | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_graphs_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.8195e-02 | 1.7188e-02 | True |
| `ae_graphs_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.8195e-02 | 1.7188e-02 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.7247e-02 | 1.7096e-02 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.7247e-02 | 1.7096e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.6384e-02 | 1.6133e-02 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.6384e-02 | 1.6133e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.7071e-02 | 1.7250e-02 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.7071e-02 | 1.7250e-02 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.8347e-02 | 1.5100e-02 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.8347e-02 | 1.5100e-02 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.6933e-02 | 1.5623e-02 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.6933e-02 | 1.5623e-02 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3751e-01 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3751e-01 | True |
| `vlm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3756e-01 | True |
| `vlm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3756e-01 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 2.2744e-02 | 3.0800e-02 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 2.2744e-02 | 3.0800e-02 | True |

## Preparations

- `ae_graphs_mpad`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 20.2431, 'ms_at_pad': 5.5272}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 942, 'ms_at_M': 0.1351, 'ms_at_pad': 0.121}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1022, 'ms_at_M': 0.3293, 'ms_at_pad': 0.2837}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0402, 'ms_at_pad': 0.0376}}
- `ae_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 20.4185, 'ms_at_pad': 5.5806}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 974, 'ms_at_M': 0.1356, 'ms_at_pad': 0.1215}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 0.3276, 'ms_at_pad': 0.2873}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0402, 'ms_at_pad': 0.0378}}
- `llm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 57 Linear modules padded; shapes padded: {'M=814 K=2048 N=6144 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.2157, 'ms_at_pad': 0.2019}, 'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 20.1701, 'ms_at_pad': 5.6921}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 974, 'ms_at_M': 0.1355, 'ms_at_pad': 0.1216}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.3577, 'ms_at_pad': 0.2948}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0413, 'ms_at_pad': 0.038}}
- `vlm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 19.7958, 'ms_at_pad': 5.5305}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': 0.1356, 'ms_at_pad': 0.122}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 0.3237, 'ms_at_pad': 0.2946}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0412, 'ms_at_pad': 0.0381}}

