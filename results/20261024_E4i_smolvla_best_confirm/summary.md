# 20261024_E4i_smolvla_best_confirm — T005, smolvla, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands` | measured | 28.19 (28.07–29.13) | 15.19 (15.15–15.33) | 4.28 (4.22–4.52) | 5.41 (5.35–5.74) | 3.35 (3.28–3.68) | 19.2 | 2.66 | 0.03 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands` | measured | 38.56 (38.43–39.37) | 15.20 (15.17–15.33) | 4.28 (4.23–4.42) | 15.74 (15.64–16.07) | 3.36 (3.30–3.57) | 40.8 | 2.56 | 0.04 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands` | measured | 48.90 (48.78–49.37) | 15.21 (15.18–15.30) | 4.29 (4.24–4.36) | 26.00 (25.91–26.18) | 3.39 (3.34–3.49) | 53.2 | 2.54 | 91.66 | 0 | False | normal->normal |

Per-step time by slope: `bf16_mpad_graphs_ae_at_islands`: 2.57 ms/step (ae_total, N = [2, 6, 10], intercept 0.27).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.2402e-02 | 1.2075e-02 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 7.3870e-02 | 1.3172e-02 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 7.3870e-02 | 1.3172e-02 | True |

## Preparations

- `bf16_mpad_graphs_ae_at_islands`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert, replayed from the recorded best_bf16 table): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': nan, 'ms_at_pad': nan}} | bf16_islands: 21 fp32 Linear modules (13.1 MB) -> bf16, input cast to bf16, output cast back

