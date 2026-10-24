# 20261025_E4i_pi05_best_search_s4 — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_graphs_ae_at_islands` | measured | 157.40 (156.89–158.32) | 20.63 (20.54–20.80) | 59.01 (58.85–59.26) | 75.17 (74.79–75.84) | 2.55 (2.51–2.68) | 47.8 | 7.44 | 84.89 | 0 | False | normal->normal |
| `bf16_graphs_ae_at_islands_kvs` | measured | 144.52 (144.02–146.05) | 20.42 (20.30–20.58) | 59.15 (58.99–59.31) | 62.41 (62.07–63.67) | 2.59 (2.52–2.73) | 43.2 | 6.20 | 84.94 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_islands_kvs` | measured | 144.08 (143.75–144.60) | 20.79 (20.70–20.92) | 60.40 (60.26–60.57) | 60.32 (60.09–60.64) | 2.55 (2.52–2.62) | 41.9 | 5.95 | 89.86 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.7669e-03 | 2.3587e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.1117e-03 | 3.0302e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.5281e-03 | 2.3616e-03 | True |
| `bf16_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.3057e-03 | 2.2875e-03 | True |
| `bf16_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.8795e-03 | 3.0260e-03 | True |
| `bf16_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.5666e-03 | 2.2944e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.9025e-03 | 2.3121e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.4490e-03 | 2.9988e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.1488e-03 | 2.3201e-03 | True |

## Preparations

- `bf16_graphs_ae_at_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back
- `bf16_graphs_ae_at_islands_kvs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back | lerobot.policies.pi05.modeling_pi05.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)
- `bf16_mpad_graphs_ae_at_islands_kvs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0874, 'ms_at_pad': 0.0785}} | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back | lerobot.policies.pi05.modeling_pi05.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)

