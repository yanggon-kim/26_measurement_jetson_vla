# 20261022_E4c_pi0_opt_ladder — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 169.57 (168.61–170.46) | — | — | — | — | — | — | 0.17 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 201.41 (200.75–202.51) | — | — | — | — | — | — | 0.2 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 272.05 (271.17–273.55) | 55.37 (55.22–55.53) | 126.79 (126.03–127.52) | 87.17 (86.76–88.04) | 2.69 (2.60–2.79) | 32.0 | 8.64 | 43.71 | 0 | False | normal->normal |
| `ae_graphs` | measured | 264.43 (262.78–265.13) | 55.32 (55.21–55.49) | 124.66 (124.09–125.25) | 81.82 (80.52–82.17) | 2.63 (2.58–2.68) | 30.9 | 8.10 | 20.08 | 0 | False | normal->normal |
| `bf16_base` | not applicable: checkpoint already loads bf16 (dominant parameter dtype bfloat16), so there is no fp32 base to cast | | | | | | | | | | | |
| `bf16_both` | measured | 217.39 (216.30–218.41) | 21.84 (21.75–22.00) | 110.43 (109.56–111.33) | 82.43 (82.12–82.94) | 2.64 (2.60–2.73) | 37.9 | 8.17 | 86.61 | 0 | False | normal->normal |
| `bf16_both_graphs` | measured | 211.71 (210.48–212.77) | 21.05 (20.92–21.24) | 108.92 (108.03–109.66) | 79.13 (78.63–79.95) | 2.56 (2.49–2.70) | 37.4 | 7.84 | 48.72 | 0 | False | normal->normal |
| `bf16_norms` | measured | 265.82 (265.03–266.97) | 54.63 (54.51–54.79) | 126.77 (126.15–127.64) | 81.77 (81.34–82.32) | 2.61 (2.57–2.67) | 30.8 | 8.10 | 21.0 | 0 | False | normal->normal |
| `bf16_vision` | measured | 238.45 (237.73–239.15) | 28.99 (28.79–29.21) | 124.67 (124.25–125.26) | 82.04 (81.63–82.64) | 2.67 (2.63–2.75) | 34.4 | 8.13 | 20.51 | 0 | False | normal->normal |
| `eager_bf16` | measured | 293.72 (292.02–295.38) | 28.82 (28.68–28.91) | 125.50 (124.77–126.34) | 136.55 (135.28–137.83) | 2.75 (2.72–2.81) | 46.5 | 13.55 | 1.12 | 0 | False | normal->normal |
| `eager_fp32` | measured | 436.55 (427.38–439.70) | 173.19 (173.10–173.32) | 126.09 (125.46–126.74) | 134.71 (125.32–137.14) | 2.79 (2.71–2.90) | 30.9 | 13.36 | 1.25 | 0 | False | normal->normal |
| `eager_tf32` | measured | 316.20 (314.75–318.68) | 55.19 (55.07–55.36) | 126.00 (125.27–126.85) | 132.31 (130.91–134.37) | 2.78 (2.73–2.83) | 41.8 | 13.14 | 1.16 | 0 | False | normal->normal |
| `vis_compile` | measured | 264.04 (263.14–265.17) | 51.78 (51.54–51.97) | 129.61 (128.78–130.32) | 79.98 (79.58–80.62) | 2.67 (2.61–2.77) | 30.3 | 7.92 | 51.01 | 0 | False | normal->normal |
| `vis_graphs` | measured | 256.96 (256.27–257.69) | 43.36 (43.25–43.47) | 129.65 (129.07–130.20) | 81.26 (80.93–81.66) | 2.64 (2.59–2.72) | 31.6 | 8.06 | 33.03 | 0 | False | normal->normal |
| `vis_static_pos` | not applicable: rung vis_static_pos applies to ['smolvla'] only | | | | | | | | | | | |
| `vlm_compile` | measured | 234.07 (232.29–235.92) | 43.37 (43.19–43.57) | 107.77 (106.94–108.41) | 80.20 (79.06–81.77) | 2.66 (2.53–2.80) | 34.3 | 7.95 | 88.08 | 0 | False | normal->normal |
| `vlm_compile_nvg` | measured | 249.23 (248.17–250.34) | 55.48 (55.33–55.74) | 109.14 (108.50–110.03) | 81.93 (81.52–82.41) | 2.66 (2.61–2.71) | 32.9 | 8.12 | 73.52 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 234.16 (233.29–235.06) | 43.14 (42.96–43.27) | 108.55 (108.01–109.24) | 79.81 (79.35–80.36) | 2.63 (2.59–2.69) | 34.1 | 7.91 | 48.1 | 0 | False | normal->normal |
| `whole_graphs` | measured | 232.95 (231.98–234.14) | — | — | — | — | — | — | 82.2 | 0 | False | normal->normal |

Per-step time by slope: `whole_graphs`: 7.92 ms/step (e2e, N = [2, 6, 10], intercept 153.78).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N10` | `eager_fp32_N10` | 2.6678e-02 | 4.7439e-03 | True |
| `ae_compile_N10` | `eager_bf16_N10` | 3.3921e-02 | 5.6955e-03 | True |
| `ae_compile_N10` | `eager_tf32_N10` | 2.6638e-02 | 4.5129e-03 | True |
| `ae_graphs_N10` | `eager_fp32_N10` | 2.4675e-02 | 4.7899e-03 | True |
| `ae_graphs_N10` | `eager_bf16_N10` | 3.2573e-02 | 5.7400e-03 | True |
| `ae_graphs_N10` | `eager_tf32_N10` | 2.3624e-02 | 4.5361e-03 | True |
| `bf16_both_N10` | `eager_fp32_N10` | 1.0893e-02 | 2.8708e-03 | True |
| `bf16_both_N10` | `eager_bf16_N10` | 1.0857e-02 | 2.7174e-03 | True |
| `bf16_both_N10` | `eager_tf32_N10` | 1.2110e-02 | 2.9626e-03 | True |
| `bf16_both_graphs_N10` | `eager_fp32_N10` | 1.1798e-02 | 3.2270e-03 | True |
| `bf16_both_graphs_N10` | `eager_bf16_N10` | 1.3011e-02 | 3.2009e-03 | True |
| `bf16_both_graphs_N10` | `eager_tf32_N10` | 1.3253e-02 | 3.4188e-03 | True |
| `bf16_norms_N10` | `eager_fp32_N10` | 2.3389e-02 | 4.0989e-03 | True |
| `bf16_norms_N10` | `eager_bf16_N10` | 2.7839e-02 | 4.9226e-03 | True |
| `bf16_norms_N10` | `eager_tf32_N10` | 2.2885e-02 | 3.9489e-03 | True |
| `bf16_vision_N10` | `eager_fp32_N10` | 2.6160e-02 | 4.7668e-03 | True |
| `bf16_vision_N10` | `eager_bf16_N10` | 3.2981e-02 | 5.7667e-03 | True |
| `bf16_vision_N10` | `eager_tf32_N10` | 2.6119e-02 | 4.6079e-03 | True |
| `eager_bf16_N10` | `eager_fp32_N10` | 9.5787e-03 | 2.5115e-03 | True |
| `eager_bf16_N10` | `eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_N10` | `eager_tf32_N10` | 1.1385e-02 | 2.6700e-03 | True |
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `eager_bf16_N10` | 9.5787e-03 | 2.5123e-03 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 8.7663e-03 | 2.0672e-03 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 8.7663e-03 | 2.0672e-03 | True |
| `eager_tf32_N10` | `eager_bf16_N10` | 1.1385e-02 | 2.6709e-03 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `vis_compile_N10` | `eager_fp32_N10` | 3.3502e-01 | 9.7926e-03 | True |
| `vis_compile_N10` | `eager_bf16_N10` | 3.3524e-01 | 1.0397e-02 | True |
| `vis_compile_N10` | `eager_tf32_N10` | 3.3498e-01 | 9.7278e-03 | True |
| `vis_graphs_N10` | `eager_fp32_N10` | 2.0676e-02 | 4.0331e-03 | True |
| `vis_graphs_N10` | `eager_bf16_N10` | 2.8574e-02 | 5.0628e-03 | True |
| `vis_graphs_N10` | `eager_tf32_N10` | 1.9625e-02 | 3.9515e-03 | True |
| `vlm_compile_N10` | `eager_fp32_N10` | 2.4111e-02 | 4.5114e-03 | True |
| `vlm_compile_N10` | `eager_bf16_N10` | 3.2009e-02 | 5.5119e-03 | True |
| `vlm_compile_N10` | `eager_tf32_N10` | 2.3060e-02 | 4.3709e-03 | True |
| `vlm_compile_nvg_N10` | `eager_fp32_N10` | 2.1676e-02 | 4.0786e-03 | True |
| `vlm_compile_nvg_N10` | `eager_bf16_N10` | 2.7829e-02 | 5.0838e-03 | True |
| `vlm_compile_nvg_N10` | `eager_tf32_N10` | 2.1173e-02 | 3.9726e-03 | True |
| `vlm_graphs_N10` | `eager_fp32_N10` | 2.4667e-02 | 4.4932e-03 | True |
| `vlm_graphs_N10` | `eager_bf16_N10` | 3.1624e-02 | 5.4423e-03 | True |
| `vlm_graphs_N10` | `eager_tf32_N10` | 2.4627e-02 | 4.3427e-03 | True |
| `whole_graphs_N10` | `eager_fp32_N10` | 2.2078e-02 | 4.1091e-03 | True |
| `whole_graphs_N10` | `eager_bf16_N10` | 2.8939e-02 | 5.0596e-03 | True |
| `whole_graphs_N10` | `eager_tf32_N10` | 2.1575e-02 | 4.0539e-03 | True |

## Preparations

- `bf16_both`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_both_graphs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_norms`: 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_vision`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed
- `eager_bf16`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)

