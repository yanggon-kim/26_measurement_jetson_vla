# 20261023_E4i_pi05_best_search_s2 — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_graphs_ae_at_islands` | measured | 163.20 (162.43–165.32) | 20.79 (20.64–21.15) | 65.02 (64.55–65.39) | 74.64 (74.22–76.58) | 2.77 (2.70–2.97) | 45.7 | 7.39 | 87.44 | 0 | False | normal->normal |
| `bf16_graphs_islands` | measured | 176.08 (175.35–178.14) | 20.78 (20.69–21.14) | 65.83 (65.20–66.60) | 86.70 (86.21–88.58) | 2.72 (2.66–2.88) | 49.2 | 8.60 | 49.67 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at` | measured | 172.51 (171.01–174.76) | 21.00 (20.85–21.30) | 66.45 (65.91–66.99) | 82.44 (81.04–83.85) | 2.71 (2.58–2.89) | 47.8 | 8.16 | 83.84 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_islands` | measured | 167.68 (166.55–170.87) | 20.96 (20.81–21.40) | 68.06 (67.38–68.73) | 75.53 (74.91–78.51) | 2.84 (2.74–3.05) | 45.0 | 7.49 | 85.73 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_vlm_at_islands` | measured | 173.33 (171.38–176.44) | 22.16 (21.95–22.42) | 68.94 (68.10–69.71) | 79.08 (77.49–81.57) | 3.06 (2.95–3.20) | 45.6 | 7.82 | 137.41 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.0833e-03 | 2.2978e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.3117e-03 | 2.9880e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.3428e-03 | 2.3015e-03 | True |
| `bf16_graphs_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 6.1129e-03 | 2.4992e-03 | True |
| `bf16_graphs_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.6340e-03 | 3.0092e-03 | True |
| `bf16_graphs_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.9038e-03 | 2.5097e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.0137e-03 | 1.9075e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.3143e-03 | 2.9091e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.2746e-03 | 1.9197e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.9154e-03 | 2.3256e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.9293e-03 | 2.9807e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 4.9728e-03 | 2.3313e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 7.0001e-03 | 2.6010e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.4775e-03 | 2.9922e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 6.6751e-03 | 2.5713e-03 | True |

## Preparations

- `bf16_graphs_ae_at_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back
- `bf16_graphs_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back
- `bf16_mpad_graphs_ae_at`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0876, 'ms_at_pad': 0.0781}}
- `bf16_mpad_graphs_ae_at_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1128, 'ms_at_M': 0.0875, 'ms_at_pad': 0.0782}} | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back
- `bf16_mpad_graphs_ae_vlm_at_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1128, 'ms_at_M': 0.0873, 'ms_at_pad': 0.079}, 'M=968 K=2048 N=256 torch.bfloat16': {'pad_to': 1000, 'ms_at_M': 0.0269, 'ms_at_pad': 0.0252}} | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back

