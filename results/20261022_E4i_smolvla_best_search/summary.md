# 20261022_E4i_smolvla_best_search — T005, smolvla, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs` | measured | 49.76 (49.57–50.39) | 15.42 (15.38–15.52) | 3.93 (3.91–4.02) | 27.21 (27.09–27.56) | 3.18 (3.13–3.38) | 54.7 | 2.66 | 50.46 | 0 | False | normal->normal |
| `bf16_mpad_graphs_autotune` | measured | 54.41 (54.23–55.23) | 16.74 (16.71–16.88) | 5.53 (5.48–5.61) | 28.98 (28.85–29.36) | 3.16 (3.12–3.35) | 53.3 | 2.83 | 97.53 | 0 | False | normal->normal |
| `bf16_mpad_graphs_nofuse` | measured | 49.16 (48.99–49.53) | 15.47 (15.44–15.52) | 3.88 (3.85–3.98) | 26.47 (26.34–26.67) | 3.33 (3.28–3.44) | 53.8 | 2.58 | 49.45 | 0 | False | normal->normal |
| `bf16_mpad_nvg` | measured | 52.00 (51.82–52.29) | 15.54 (15.50–15.60) | 5.30 (5.24–5.45) | 27.77 (27.66–27.95) | 3.36 (3.31–3.43) | 53.4 | 2.71 | 68.41 | 0 | False | normal->normal |
| `bf16_mpad_nvg_autotune` | measured | 51.68 (51.46–52.17) | 16.17 (16.13–16.25) | 5.21 (5.16–5.37) | 26.94 (26.80–27.26) | 3.35 (3.29–3.45) | 52.1 | 2.63 | 113.83 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs` | measured | 47.94 (47.87–48.07) | — | — | — | — | — | — | 218.54 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs_autotune` | measured | 46.55 (46.44–46.70) | — | — | — | — | — | — | 298.77 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.7791e-02 | 9.7400e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 8.2917e-02 | 1.5901e-02 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 8.2917e-02 | 1.5901e-02 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.9760e-02 | 8.4482e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 8.4442e-02 | 1.6122e-02 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 8.4442e-02 | 1.6122e-02 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 5.2381e-02 | 1.0139e-02 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 9.8952e-02 | 1.5882e-02 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 9.8952e-02 | 1.5882e-02 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.9210e-02 | 1.0309e-02 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 7.7850e-02 | 1.3844e-02 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 7.7850e-02 | 1.3844e-02 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.2912e-02 | 1.1163e-02 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 9.0653e-02 | 1.7129e-02 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 9.0653e-02 | 1.7129e-02 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 3.6442e-02 | 9.1435e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 6.1988e-02 | 1.0867e-02 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 6.1988e-02 | 1.0867e-02 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.5534e-02 | 1.2903e-02 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 8.8685e-02 | 1.8839e-02 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 8.8685e-02 | 1.8839e-02 | True |

## Preparations

- `bf16_mpad_graphs`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 32 Linear modules padded; shapes padded: {'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0253, 'ms_at_pad': 0.023}}
- `bf16_mpad_graphs_autotune`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0224, 'ms_at_pad': 0.0206}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0257, 'ms_at_pad': 0.0236}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.0322, 'ms_at_pad': 0.0287}}
- `bf16_mpad_graphs_nofuse`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 32 Linear modules padded; shapes padded: {'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0259, 'ms_at_pad': 0.0235}} | Inductor's SDPA pattern rewrite (fuse_attention._sfdp_init) disabled
- `bf16_mpad_nvg`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0216, 'ms_at_pad': 0.0202}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0256, 'ms_at_pad': 0.0236}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0321, 'ms_at_pad': 0.0301}}
- `bf16_mpad_nvg_autotune`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 225, 'ms_at_M': 0.0217, 'ms_at_pad': 0.0205}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0239}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0317, 'ms_at_pad': 0.0291}}
- `bf16_mpad_whole_graphs`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0215, 'ms_at_pad': 0.0203}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0258, 'ms_at_pad': 0.0245}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.032, 'ms_at_pad': 0.03}}
- `bf16_mpad_whole_graphs_autotune`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0228, 'ms_at_pad': 0.0208}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0237}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.0311, 'ms_at_pad': 0.0294}}

