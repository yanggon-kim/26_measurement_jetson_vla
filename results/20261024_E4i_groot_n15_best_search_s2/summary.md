# 20261024_E4i_groot_n15_best_search_s2 — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 124.06 (116.37–127.71) | 18.59 (18.53–18.73) | 30.86 (30.73–31.16) | 36.73 (36.62–37.09) | 37.81 (30.04–40.02) | 107.46 (107.14–108.37) | 34.2 | — | 158.07 | 0 | False | normal->normal |
| `all_compile_ae_at` | measured | 109.32 (101.46–112.66) | 12.15 (12.04–12.29) | 23.04 (22.83–23.53) | 36.05 (35.92–36.24) | 38.09 (30.26–40.78) | 92.99 (92.65–94.14) | 38.8 | — | 180.41 | 0 | False | normal->normal |
| `all_compile_ae_at_mpad` | measured | 97.57 (89.54–100.43) | 12.26 (12.16–12.36) | 23.43 (23.21–23.79) | 37.89 (37.79–38.09) | 23.95 (15.92–25.93) | 80.90 (80.53–81.40) | 46.8 | — | 173.59 | 0 | False | normal->normal |
| `all_compile_ae_vis_at` | measured | 112.14 (111.11–116.36) | 12.76 (12.63–12.96) | 23.68 (23.34–24.57) | 37.12 (37.03–37.30) | 38.52 (37.59–42.39) | 95.56 (94.86–96.41) | 38.8 | — | 205.44 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 107.85 (101.46–109.86) | 12.14 (12.01–12.23) | 21.79 (21.66–22.41) | 36.03 (35.94–36.32) | 37.84 (30.50–39.68) | 91.57 (91.23–92.36) | 39.4 | — | 198.04 | 0 | False | normal->normal |
| `llm_compile_ae_at` | measured | 116.78 (112.92–118.66) | 19.06 (18.97–19.24) | 23.64 (23.33–24.23) | 35.49 (35.40–35.74) | 38.61 (35.15–39.91) | 99.97 (99.51–100.93) | 35.5 | — | 164.36 | 0 | False | normal->normal |
| `llm_graphs_mpad_autotune` | measured | 101.54 (100.02–104.89) | 19.04 (18.84–19.26) | 17.67 (17.52–17.98) | 40.94 (40.84–41.13) | 23.89 (22.82–27.10) | 84.82 (84.49–85.92) | 48.3 | — | 139.49 | 0 | True | normal->normal |
| `vlm_graphs_ae_at` | measured | 111.40 (103.49–113.17) | 12.38 (12.29–12.55) | 23.87 (23.63–24.16) | 36.20 (36.10–36.40) | 38.98 (30.99–40.59) | 94.63 (94.17–95.56) | 38.3 | — | 182.92 | 0 | False | normal->normal |
| `vlm_graphs_mpad_autotune` | measured | 92.41 (91.27–94.43) | 12.45 (12.32–12.64) | 18.45 (18.29–18.95) | 37.65 (37.53–37.88) | 23.79 (22.95–25.33) | 75.79 (75.53–76.83) | 49.7 | — | 185.41 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.8681e-02 | 1.8262e-02 | True |
| `ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.8681e-02 | 1.8262e-02 | True |
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 2.9032e-02 | 3.5904e-02 | True |
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 2.9032e-02 | 3.5904e-02 | True |
| `all_compile_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3728e-01 | True |
| `all_compile_ae_at_mpad_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3728e-01 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3904e-01 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3904e-01 | True |
| `all_compile_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3850e-01 | True |
| `all_compile_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3850e-01 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3680e-01 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3680e-01 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.3767e-01 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.3767e-01 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 2.9699e-02 | 3.2603e-02 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 2.9699e-02 | 3.2603e-02 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 1.0000e+00 | 1.4087e-01 | True |
| `vlm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 1.0000e+00 | 1.4087e-01 | True |

## Preparations

- `all_compile_ae_at_mpad`: linear_mpad(backbone+action_head): 33 Linear modules padded; shapes padded: {'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 19.9005, 'ms_at_pad': 5.3268}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 974, 'ms_at_M': 0.1354, 'ms_at_pad': 0.1213}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1022, 'ms_at_M': 0.3283, 'ms_at_pad': 0.2845}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0409, 'ms_at_pad': 0.0376}}
- `llm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 73 Linear modules padded; shapes padded: {'M=814 K=2048 N=6144 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.2217, 'ms_at_pad': 0.2057}, 'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 1006, 'ms_at_M': 19.5055, 'ms_at_pad': 5.7128}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': 0.1354, 'ms_at_pad': 0.1213}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 0.3422, 'ms_at_pad': 0.2894}, 'M=49 K=1536 N=1536 torch.bfloat16': {'pad_to': 129, 'ms_at_M': 0.0223, 'ms_at_pad': 0.0211}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0411, 'ms_at_pad': 0.0377}}
- `vlm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 81 Linear modules padded; shapes padded: {'M=814 K=2048 N=1024 torch.bfloat16': {'pad_to': 958, 'ms_at_M': 0.0479, 'ms_at_pad': 0.0424}, 'M=814 K=2048 N=6144 torch.bfloat16': {'pad_to': 974, 'ms_at_M': 0.2223, 'ms_at_pad': 0.21}, 'M=814 K=2048 N=151680 torch.bfloat16': {'pad_to': 990, 'ms_at_M': 19.9428, 'ms_at_pad': 5.7204}, 'M=814 K=2048 N=2048 torch.float32': {'pad_to': 910, 'ms_at_M': 0.136, 'ms_at_pad': 0.1215}, 'M=814 K=8192 N=2048 torch.bfloat16': {'pad_to': 1022, 'ms_at_M': 0.3332, 'ms_at_pad': 0.2835}, 'M=49 K=6144 N=1536 torch.bfloat16': {'pad_to': 145, 'ms_at_M': 0.0395, 'ms_at_pad': 0.037}}

