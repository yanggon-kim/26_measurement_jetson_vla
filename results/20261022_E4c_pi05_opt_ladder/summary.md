# 20261022_E4c_pi05_opt_ladder — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 227.75 (225.82–230.02) | — | — | — | — | — | — | 0.23 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 282.50 (279.68–284.93) | — | — | — | — | — | — | 0.28 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 384.71 (380.28–387.64) | 55.03 (54.89–55.23) | 183.32 (179.17–186.02) | 143.44 (142.90–144.07) | 2.80 (2.73–2.99) | 37.3 | 14.26 | 39.55 | 0 | False | normal->normal |
| `ae_graphs` | measured | 376.83 (373.50–380.11) | 55.24 (55.02–55.47) | 183.18 (179.65–185.98) | 135.35 (134.55–136.91) | 2.88 (2.79–2.97) | 35.9 | 13.48 | 21.1 | 0 | False | normal->normal |
| `bf16_base` | measured | 247.48 (246.71–248.49) | 55.59 (55.45–55.77) | 96.52 (95.82–97.24) | 92.56 (92.21–93.18) | 2.75 (2.70–2.86) | 37.4 | 9.18 | 21.0 | 0 | False | normal->normal |
| `bf16_both` | measured | 186.77 (185.53–188.48) | 21.38 (21.24–21.64) | 68.91 (67.94–69.63) | 93.62 (93.32–94.89) | 2.80 (2.74–2.97) | 50.1 | 9.29 | 80.32 | 0 | False | normal->normal |
| `bf16_both_graphs` | measured | 183.08 (182.05–184.13) | 21.13 (21.02–21.35) | 66.88 (66.09–67.65) | 92.43 (91.93–92.99) | 2.63 (2.55–2.75) | 50.5 | 9.17 | 50.69 | 0 | False | normal->normal |
| `bf16_norms` | measured | 249.91 (248.51–252.10) | 55.19 (55.04–55.38) | 98.94 (98.19–99.77) | 92.76 (92.09–94.47) | 2.75 (2.69–2.94) | 37.1 | 9.23 | 21.32 | 0 | False | normal->normal |
| `bf16_vision` | measured | 222.31 (221.37–228.61) | 29.57 (29.42–29.97) | 95.64 (95.19–96.41) | 94.20 (93.66–98.52) | 2.83 (2.73–3.01) | 42.4 | 9.35 | 21.63 | 0 | False | normal->normal |
| `eager_bf16` | measured | 277.36 (275.98–278.98) | 29.22 (29.07–29.36) | 96.26 (95.75–96.81) | 149.04 (147.76–150.26) | 2.82 (2.75–2.90) | 53.7 | 14.78 | 1.12 | 0 | False | normal->normal |
| `eager_fp32` | measured | 1427.45 (1425.72–1430.90) | 174.76 (174.63–174.89) | 854.31 (853.98–854.61) | 395.63 (393.75–398.69) | 2.88 (2.78–3.03) | 27.7 | 39.49 | 2.17 | 0 | False | normal->normal |
| `eager_tf32` | measured | 399.79 (397.36–402.18) | 55.41 (55.23–55.59) | 163.90 (162.19–165.62) | 177.47 (176.64–179.58) | 2.70 (2.65–2.78) | 44.4 | 17.70 | 1.16 | 0 | False | normal->normal |
| `vis_compile` | measured | 378.54 (375.36–380.63) | 48.69 (48.48–48.96) | 188.65 (185.92–190.55) | 138.39 (137.83–139.03) | 2.82 (2.76–2.89) | 36.6 | 13.77 | 46.25 | 0 | False | normal->normal |
| `vis_graphs` | measured | 373.30 (370.84–376.83) | 43.00 (42.83–43.17) | 193.83 (191.28–195.90) | 133.43 (132.88–136.18) | 2.77 (2.69–2.96) | 35.7 | 13.30 | 33.16 | 0 | False | normal->normal |
| `vis_static_pos` | not applicable: rung vis_static_pos applies to ['smolvla'] only | | | | | | | | | | | |
| `vlm_compile` | measured | 317.14 (315.19–319.21) | 43.46 (43.34–43.69) | 137.79 (136.04–139.76) | 133.04 (132.62–133.94) | 2.73 (2.66–2.86) | 41.9 | 13.24 | 65.55 | 0 | False | normal->normal |
| `vlm_compile_nvg` | measured | 344.08 (341.02–346.97) | 55.27 (55.12–55.50) | 153.43 (150.96–155.68) | 132.49 (131.68–133.82) | 2.78 (2.71–2.90) | 38.5 | 13.18 | 52.38 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 333.29 (329.82–336.29) | 43.24 (43.13–43.46) | 154.20 (151.42–156.78) | 132.60 (131.98–134.29) | 2.69 (2.58–2.80) | 39.8 | 13.23 | 48.26 | 0 | False | normal->normal |
| `whole_graphs` | measured | 335.00 (332.66–337.62) | — | — | — | — | — | — | 80.01 | 0 | False | normal->normal |

Per-step time by slope: `whole_graphs`: 13.41 ms/step (e2e, N = [2, 6, 10], intercept 201.31).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N10` | `eager_fp32_N10` | 5.5850e-04 | 2.0931e-04 | True |
| `ae_compile_N10` | `eager_bf16_N10` | 7.5436e-03 | 2.7299e-03 | True |
| `ae_compile_N10` | `eager_tf32_N10` | 9.2345e-04 | 1.7525e-04 | True |
| `ae_graphs_N10` | `eager_fp32_N10` | 5.5606e-04 | 2.1217e-04 | True |
| `ae_graphs_N10` | `eager_bf16_N10` | 7.4440e-03 | 2.7296e-03 | True |
| `ae_graphs_N10` | `eager_tf32_N10` | 1.0012e-03 | 1.9482e-04 | True |
| `bf16_base_N10` | `eager_fp32_N10` | 5.2210e-03 | 1.8733e-03 | True |
| `bf16_base_N10` | `eager_bf16_N10` | 7.8299e-03 | 2.7652e-03 | True |
| `bf16_base_N10` | `eager_tf32_N10` | 5.4819e-03 | 1.8877e-03 | True |
| `bf16_both_N10` | `eager_fp32_N10` | 3.5511e-03 | 2.2015e-03 | True |
| `bf16_both_N10` | `eager_bf16_N10` | 8.0464e-03 | 2.6858e-03 | True |
| `bf16_both_N10` | `eager_tf32_N10` | 3.5230e-03 | 2.1958e-03 | True |
| `bf16_both_graphs_N10` | `eager_fp32_N10` | 4.2580e-03 | 1.8753e-03 | True |
| `bf16_both_graphs_N10` | `eager_bf16_N10` | 9.4394e-03 | 2.7488e-03 | True |
| `bf16_both_graphs_N10` | `eager_tf32_N10` | 3.8416e-03 | 1.8782e-03 | True |
| `bf16_norms_N10` | `eager_fp32_N10` | 5.2210e-03 | 1.8733e-03 | True |
| `bf16_norms_N10` | `eager_bf16_N10` | 7.8299e-03 | 2.7652e-03 | True |
| `bf16_norms_N10` | `eager_tf32_N10` | 5.4819e-03 | 1.8877e-03 | True |
| `bf16_vision_N10` | `eager_fp32_N10` | 5.2604e-03 | 2.1494e-03 | True |
| `bf16_vision_N10` | `eager_bf16_N10` | 6.4793e-03 | 2.6057e-03 | True |
| `bf16_vision_N10` | `eager_tf32_N10` | 4.9355e-03 | 2.1308e-03 | True |
| `eager_bf16_N10` | `eager_fp32_N10` | 7.6400e-03 | 2.7309e-03 | True |
| `eager_bf16_N10` | `eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_N10` | `eager_tf32_N10` | 7.3837e-03 | 2.7257e-03 | True |
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `eager_bf16_N10` | 7.6400e-03 | 2.7323e-03 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 5.7006e-04 | 2.1135e-04 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 5.7006e-04 | 2.1135e-04 | True |
| `eager_tf32_N10` | `eager_bf16_N10` | 7.3837e-03 | 2.7272e-03 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `vis_compile_N10` | `eager_fp32_N10` | 5.0819e-04 | 2.0621e-04 | True |
| `vis_compile_N10` | `eager_bf16_N10` | 7.3821e-03 | 2.7231e-03 | True |
| `vis_compile_N10` | `eager_tf32_N10` | 1.0281e-03 | 1.8597e-04 | True |
| `vis_graphs_N10` | `eager_fp32_N10` | 5.7077e-04 | 2.0077e-04 | True |
| `vis_graphs_N10` | `eager_bf16_N10` | 7.6005e-03 | 2.7320e-03 | True |
| `vis_graphs_N10` | `eager_tf32_N10` | 9.4306e-04 | 1.9502e-04 | True |
| `vlm_compile_N10` | `eager_fp32_N10` | 5.7871e-04 | 2.1919e-04 | True |
| `vlm_compile_N10` | `eager_bf16_N10` | 7.5614e-03 | 2.7236e-03 | True |
| `vlm_compile_N10` | `eager_tf32_N10` | 1.0409e-03 | 1.9491e-04 | True |
| `vlm_compile_nvg_N10` | `eager_fp32_N10` | 5.6696e-04 | 2.0002e-04 | True |
| `vlm_compile_nvg_N10` | `eager_bf16_N10` | 7.6540e-03 | 2.7348e-03 | True |
| `vlm_compile_nvg_N10` | `eager_tf32_N10` | 9.4378e-04 | 1.8679e-04 | True |
| `vlm_graphs_N10` | `eager_fp32_N10` | 5.8782e-04 | 2.1228e-04 | True |
| `vlm_graphs_N10` | `eager_bf16_N10` | 7.5112e-03 | 2.7389e-03 | True |
| `vlm_graphs_N10` | `eager_tf32_N10` | 9.3055e-04 | 1.9957e-04 | True |
| `whole_graphs_N10` | `eager_fp32_N10` | 5.8436e-04 | 2.1411e-04 | True |
| `whole_graphs_N10` | `eager_bf16_N10` | 7.4620e-03 | 2.7249e-03 | True |
| `whole_graphs_N10` | `eager_tf32_N10` | 9.0969e-04 | 1.8210e-04 | True |

## Preparations

- `bf16_base`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32)
- `bf16_both`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only)
- `bf16_both_graphs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only)
- `bf16_norms`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); 37 layer-norm modules -> bf16 (own parameters only)
- `bf16_vision`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed
- `eager_bf16`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only)

