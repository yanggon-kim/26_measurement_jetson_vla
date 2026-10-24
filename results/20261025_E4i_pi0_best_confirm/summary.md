# 20261025_E4i_pi0_best_confirm — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_kvs` | measured | 88.05 (87.90–88.27) | 20.57 (20.51–20.66) | 54.83 (54.70–54.99) | 10.27 (10.20–10.35) | 2.38 (2.35–2.42) | 11.7 | 5.09 | 0.09 | 0 | False | normal->normal |

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_kvs` | measured | 108.29 (108.09–108.50) | 20.48 (20.41–20.59) | 54.85 (54.73–55.01) | 30.56 (30.46–30.67) | 2.38 (2.35–2.43) | 28.2 | 5.02 | 0.11 | 0 | False | normal->normal |

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_kvs` | measured | 129.51 (129.24–129.85) | 20.56 (20.51–20.67) | 54.97 (54.82–55.10) | 51.46 (51.33–51.69) | 2.50 (2.47–2.56) | 39.7 | 5.07 | 89.03 | 0 | False | normal->normal |

Per-step time by slope: `bf16_mpad_graphs_ae_at_kvs`: 5.15 ms/step (ae_total, N = [2, 6, 10], intercept -0.13).

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.9522e-02 | 4.0990e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.7420e-02 | 4.8258e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.8471e-02 | 3.9388e-03 | True |

## Preparations

- `bf16_mpad_graphs_ae_at_kvs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert, replayed from the recorded best_bf16 table): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': nan, 'ms_at_pad': nan}} | lerobot.policies.pi0.modeling_pi0.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)

