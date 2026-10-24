# 20261025_E4i_pi0_best_search_s4 — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 137.98 (137.51–138.56) | 20.27 (20.20–20.40) | 55.00 (54.89–55.21) | 60.22 (59.84–60.64) | 2.47 (2.43–2.55) | 43.6 | 5.95 | 81.67 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_islands_kvs` | measured | 132.78 (132.59–133.25) | 20.22 (20.14–20.36) | 59.71 (59.60–59.85) | 50.31 (50.15–50.51) | 2.54 (2.51–2.59) | 37.9 | 4.95 | 83.09 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_kvs` | measured | 128.45 (128.25–128.87) | 20.37 (20.29–20.49) | 55.33 (55.21–55.48) | 50.25 (50.08–50.51) | 2.51 (2.47–2.56) | 39.1 | 4.95 | 92.16 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.4123e-02 | 3.0878e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.7213e-02 | 3.6443e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.1771e-02 | 3.0518e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.5461e-02 | 3.9234e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.6827e-02 | 4.3847e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.3540e-02 | 3.7347e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.9117e-02 | 3.9583e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.2141e-02 | 4.7013e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.8613e-02 | 3.8041e-03 | True |

## Preparations

- `bf16_mpad_graphs_ae_at`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0249, 'ms_at_pad': 0.0219}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.4701, 'ms_at_pad': 0.5756}}
- `bf16_mpad_graphs_ae_at_islands_kvs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0253, 'ms_at_pad': 0.0216}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1008, 'ms_at_M': 1.4918, 'ms_at_pad': 0.5787}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 848, 'ms_at_M': 1.1889, 'ms_at_pad': 1.0715}} | bf16_islands: 5 fp32 Linear modules (13.0 MB) -> bf16, input cast to bf16, output cast back | lerobot.policies.pi0.modeling_pi0.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)
- `bf16_mpad_graphs_ae_at_kvs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0253, 'ms_at_pad': 0.0217}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.4732, 'ms_at_pad': 0.5761}} | lerobot.policies.pi0.modeling_pi0.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)

