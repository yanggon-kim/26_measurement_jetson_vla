# 20261025_E4i_pi05_best_confirm — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands_kvs` | measured | 94.87 (94.72–95.08) | 20.34 (20.29–20.44) | 59.97 (59.85–60.12) | 12.08 (11.98–12.19) | 2.48 (2.45–2.55) | 12.7 | 5.98 | 0.1 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands_kvs` | measured | 118.61 (106.47–118.91) | 20.31 (17.48–20.45) | 59.90 (52.21–60.13) | 35.83 (34.19–36.04) | 2.50 (2.46–2.56) | 30.2 | 5.89 | 0.12 | 0 | False | normal->fast |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands_kvs` | measured | 142.44 (142.17–142.94) | 20.33 (20.27–20.43) | 59.98 (59.87–60.22) | 59.58 (59.35–59.88) | 2.51 (2.47–2.57) | 41.8 | 5.89 | 82.38 | 0 | False | normal->normal |

Per-step time by slope: `bf16_mpad_graphs_ae_at_islands_kvs`: 5.94 ms/step (ae_total, N = [2, 6, 10], intercept 0.20).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.0845e-03 | 2.3218e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.8090e-03 | 3.0138e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.0114e-03 | 2.3298e-03 | True |

## Preparations

- `bf16_mpad_graphs_ae_at_islands_kvs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert, replayed from the recorded best_bf16 table): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': nan, 'ms_at_pad': nan}} | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back | lerobot.policies.pi05.modeling_pi05.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)

