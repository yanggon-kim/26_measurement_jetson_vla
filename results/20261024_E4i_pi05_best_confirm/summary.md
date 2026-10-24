# 20261024_E4i_pi05_best_confirm — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_graphs_ae_at_islands` | measured | 105.27 (104.73–106.35) | 21.43 (21.32–21.70) | 65.89 (65.48–66.45) | 15.27 (15.10–15.88) | 2.63 (2.57–2.80) | 14.5 | 7.60 | 0.11 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_graphs_ae_at_islands` | measured | 134.80 (134.06–136.00) | 21.08 (20.99–21.51) | 65.89 (65.33–66.39) | 45.11 (44.80–46.15) | 2.68 (2.61–2.89) | 33.5 | 7.45 | 0.14 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_graphs_ae_at_islands` | measured | 164.39 (163.44–165.55) | 21.07 (20.97–21.34) | 65.94 (65.31–66.49) | 74.68 (74.13–76.21) | 2.70 (2.63–2.84) | 45.4 | 7.39 | 86.85 | 0 | False | normal->normal |

Per-step time by slope: `bf16_graphs_ae_at_islands`: 7.43 ms/step (ae_total, N = [2, 6, 10], intercept 0.46).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.5732e-03 | 2.3302e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.4850e-03 | 2.9905e-03 | True |
| `bf16_graphs_ae_at_islands_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.3344e-03 | 2.3299e-03 | True |

## Preparations

- `bf16_graphs_ae_at_islands`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | bf16_islands: 41 fp32 Linear modules (474.7 MB) -> bf16, input cast to bf16, output cast back

