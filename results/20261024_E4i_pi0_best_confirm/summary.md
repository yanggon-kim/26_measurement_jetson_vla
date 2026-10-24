# 20261024_E4i_pi0_best_confirm — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 99.63 (99.02–100.70) | 21.32 (21.22–21.57) | 63.05 (62.29–63.71) | 12.75 (12.63–13.21) | 2.44 (2.41–2.57) | 12.8 | 6.35 | 0.1 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 124.39 (123.64–126.07) | 21.09 (20.96–21.40) | 63.15 (62.55–63.92) | 37.63 (37.30–38.37) | 2.47 (2.42–2.70) | 30.2 | 6.20 | 0.13 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 149.35 (148.50–151.48) | 21.06 (20.94–21.42) | 63.08 (62.31–63.81) | 62.62 (62.05–63.98) | 2.52 (2.44–2.82) | 41.9 | 6.19 | 90.17 | 0 | False | normal->normal |

Per-step time by slope: `bf16_mpad_graphs_ae_at`: 6.23 ms/step (ae_total, N = [2, 6, 10], intercept 0.26).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.5129e-02 | 3.2307e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.8935e-02 | 3.7716e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.2555e-02 | 3.1928e-03 | True |

## Preparations

- `bf16_mpad_graphs_ae_at`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert, replayed from the recorded best_bf16 table): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': nan, 'ms_at_pad': nan}}

