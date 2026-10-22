# 20261022_E4c_smolvla_opt_ladder — T005, smolvla, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 25.93 (25.83–26.41) | — | — | — | — | — | — | 70.36 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 35.24 (35.12–35.46) | — | — | — | — | — | — | 138.29 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 73.36 (72.51–75.50) | 15.05 (14.97–15.36) | 14.61 (14.38–15.25) | 40.15 (39.67–41.34) | 3.48 (3.38–3.81) | 54.7 | 3.93 | 40.42 | 0 | False | normal->normal |
| `ae_graphs` | measured | 59.69 (59.22–60.48) | 14.96 (14.86–15.16) | 14.84 (14.63–15.28) | 26.47 (26.32–26.72) | 3.37 (3.32–3.51) | 44.4 | 2.58 | 21.61 | 0 | False | normal->normal |
| `bf16_base` | not applicable: checkpoint already loads bf16 (dominant parameter dtype bfloat16), so there is no fp32 base to cast | | | | | | | | | | | |
| `bf16_both` | not applicable: SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already | | | | | | | | | | | |
| `bf16_both_graphs` | not applicable: SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already | | | | | | | | | | | |
| `bf16_norms` | not applicable: SmolVLAPolicy has no params_to_keep_float32: its norms are not held in fp32 | | | | | | | | | | | |
| `bf16_vision` | not applicable: SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already | | | | | | | | | | | |
| `eager_bf16` | measured | 166.47 (165.37–170.45) | 15.01 (14.90–15.17) | 14.13 (13.88–14.58) | 133.98 (133.02–136.02) | 3.35 (3.27–3.53) | 80.5 | 13.31 | 0.75 | 0 | False | normal->normal |
| `eager_fp32` | measured | 170.06 (167.86–171.87) | 15.05 (14.92–15.24) | 14.29 (13.98–14.59) | 137.33 (135.24–138.34) | 3.47 (3.36–3.64) | 80.8 | 13.63 | 0.74 | 0 | False | normal->normal |
| `eager_tf32` | measured | 168.60 (167.44–176.01) | 14.98 (14.84–15.22) | 14.65 (14.25–15.26) | 135.78 (134.79–142.21) | 3.45 (3.32–3.60) | 80.5 | 13.50 | 0.75 | 0 | False | normal->normal |
| `vis_compile` | measured | 62.54 (62.05–63.70) | 18.21 (18.10–18.55) | 14.45 (14.22–14.85) | 26.54 (26.38–26.85) | 3.34 (3.26–3.58) | 42.4 | 2.59 | 57.66 | 0 | False | normal->normal |
| `vis_graphs` | measured | 59.34 (58.75–60.73) | 15.22 (15.15–15.41) | 14.65 (14.34–15.20) | 26.15 (25.97–26.56) | 3.31 (3.22–3.62) | 44.1 | 2.55 | 30.64 | 0 | False | normal->normal |
| `vis_static_pos` | measured | 62.72 (62.28–63.83) | 17.08 (17.03–17.27) | 14.79 (14.57–15.28) | 27.52 (27.38–28.03) | 3.32 (3.24–3.53) | 43.9 | 2.69 | 62.34 | 0 | False | normal->normal |
| `vlm_compile` | measured | 49.29 (49.00–50.63) | 15.51 (15.45–15.76) | 4.49 (4.42–4.73) | 26.03 (25.85–26.46) | 3.25 (3.16–3.68) | 52.8 | 2.55 | 64.64 | 0 | False | normal->normal |
| `vlm_compile_nvg` | measured | 49.96 (49.72–50.96) | 14.95 (14.86–15.10) | 4.23 (4.16–4.44) | 27.57 (27.41–27.99) | 3.22 (3.13–3.47) | 55.2 | 2.70 | 56.21 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 49.83 (49.30–50.99) | 15.42 (15.33–15.65) | 3.73 (3.66–3.83) | 27.39 (27.12–27.98) | 3.30 (3.12–3.60) | 55.0 | 2.68 | 50.17 | 0 | False | normal->normal |
| `whole_graphs` | measured | 46.85 (46.73–47.30) | — | — | — | — | — | — | 215.02 | 0 | False | normal->normal |

Per-step time by slope: `whole_graphs`: 2.62 ms/step (e2e, N = [2, 6, 10], intercept 20.32).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N10` | `eager_fp32_N10` | 5.2988e-02 | 1.1390e-02 | True |
| `ae_compile_N10` | `eager_bf16_N10` | 1.2385e-02 | 2.1286e-03 | True |
| `ae_compile_N10` | `eager_tf32_N10` | 1.2385e-02 | 2.1286e-03 | True |
| `ae_graphs_N10` | `eager_fp32_N10` | 5.3842e-02 | 1.1513e-02 | True |
| `ae_graphs_N10` | `eager_bf16_N10` | 1.4479e-02 | 2.0531e-03 | True |
| `ae_graphs_N10` | `eager_tf32_N10` | 1.4479e-02 | 2.0531e-03 | True |
| `eager_bf16_N10` | `eager_fp32_N10` | 5.8766e-02 | 1.1935e-02 | True |
| `eager_bf16_N10` | `eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `eager_bf16_N10` | 5.8766e-02 | 1.1847e-02 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 5.8766e-02 | 1.1847e-02 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 5.8766e-02 | 1.1935e-02 | True |
| `eager_tf32_N10` | `eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `vis_compile_N10` | `eager_fp32_N10` | 3.7654e-02 | 1.2918e-02 | True |
| `vis_compile_N10` | `eager_bf16_N10` | 3.5239e-02 | 1.0436e-02 | True |
| `vis_compile_N10` | `eager_tf32_N10` | 3.5239e-02 | 1.0436e-02 | True |
| `vis_graphs_N10` | `eager_fp32_N10` | 2.3910e-02 | 7.4448e-03 | True |
| `vis_graphs_N10` | `eager_bf16_N10` | 5.9700e-02 | 1.3259e-02 | True |
| `vis_graphs_N10` | `eager_tf32_N10` | 5.9700e-02 | 1.3259e-02 | True |
| `vis_static_pos_N10` | `eager_fp32_N10` | 3.6043e-02 | 1.1081e-02 | True |
| `vis_static_pos_N10` | `eager_bf16_N10` | 4.6206e-02 | 1.2851e-02 | True |
| `vis_static_pos_N10` | `eager_tf32_N10` | 4.6206e-02 | 1.2851e-02 | True |
| `vlm_compile_N10` | `eager_fp32_N10` | 5.0194e-02 | 1.0964e-02 | True |
| `vlm_compile_N10` | `eager_bf16_N10` | 8.2935e-02 | 1.5684e-02 | True |
| `vlm_compile_N10` | `eager_tf32_N10` | 8.2935e-02 | 1.5684e-02 | True |
| `vlm_compile_nvg_N10` | `eager_fp32_N10` | 3.9721e-02 | 1.0507e-02 | True |
| `vlm_compile_nvg_N10` | `eager_bf16_N10` | 8.1725e-02 | 1.3197e-02 | True |
| `vlm_compile_nvg_N10` | `eager_tf32_N10` | 8.1725e-02 | 1.3197e-02 | True |
| `vlm_graphs_N10` | `eager_fp32_N10` | 4.0772e-02 | 1.0475e-02 | True |
| `vlm_graphs_N10` | `eager_bf16_N10` | 7.6702e-02 | 1.1778e-02 | True |
| `vlm_graphs_N10` | `eager_tf32_N10` | 7.6702e-02 | 1.1778e-02 | True |
| `whole_graphs_N10` | `eager_fp32_N10` | 4.0530e-02 | 9.7569e-03 | True |
| `whole_graphs_N10` | `eager_bf16_N10` | 8.0434e-02 | 1.6873e-02 | True |
| `whole_graphs_N10` | `eager_tf32_N10` | 8.0434e-02 | 1.6873e-02 | True |

## Preparations

- `eager_bf16`: no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model
- `vis_graphs`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids
- `vis_static_pos`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids
- `vlm_compile`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids
- `vlm_graphs`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids
- `whole_graphs`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids

