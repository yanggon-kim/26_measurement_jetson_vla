# 20261023_E4i_smolvla_best_search_s2 — T005, smolvla, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 51.03 (50.82–51.40) | 15.41 (15.34–15.53) | 4.62 (4.56–4.72) | 27.45 (27.35–27.66) | 3.55 (3.47–3.65) | 53.8 | 2.68 | 106.39 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_islands` | measured | 49.77 (49.50–51.17) | 15.30 (15.24–15.52) | 4.33 (4.24–4.63) | 26.98 (26.79–27.66) | 3.21 (3.13–3.60) | 54.2 | 2.64 | 100.05 | 0 | False | normal->normal |
| `bf16_mpad_graphs_islands` | measured | 50.86 (50.66–51.51) | 15.64 (15.58–15.81) | 4.60 (4.56–4.70) | 27.04 (26.87–27.31) | 3.58 (3.51–3.74) | 53.2 | 2.64 | 54.57 | 0 | False | normal->normal |
| `bf16_mpad_graphs_nofuse_islands` | measured | 54.61 (53.40–56.19) | 15.57 (15.38–15.87) | 4.80 (4.71–4.95) | 30.45 (29.74–31.51) | 3.73 (3.48–3.99) | 55.8 | 2.97 | 62.8 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 5.1361e-02 | 8.6079e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 8.3403e-02 | 1.4406e-02 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 8.3403e-02 | 1.4406e-02 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.1400e-02 | 1.2030e-02 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 7.2096e-02 | 1.3207e-02 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 7.2096e-02 | 1.3207e-02 | True |
| `bf16_mpad_graphs_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 3.6794e-02 | 8.3349e-03 | True |
| `bf16_mpad_graphs_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 5.8863e-02 | 1.1576e-02 | True |
| `bf16_mpad_graphs_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 5.8863e-02 | 1.1576e-02 | True |
| `bf16_mpad_graphs_nofuse_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_fp32_N10` | 4.2817e-02 | 7.3841e-03 | True |
| `bf16_mpad_graphs_nofuse_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 8.6137e-02 | 1.4411e-02 | True |
| `bf16_mpad_graphs_nofuse_islands_N10` | `20261022_E4c_smolvla_opt_ladder:eager_tf32_N10` | 8.6137e-02 | 1.4411e-02 | True |

## Preparations

- `bf16_mpad_graphs_ae_at`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 128 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0212, 'ms_at_pad': 0.0112}, 'M=177 K=960 N=320 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.0197, 'ms_at_pad': 0.01}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 369, 'ms_at_M': 0.0259, 'ms_at_pad': 0.0179}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0199, 'ms_at_pad': 0.018}, 'M=177 K=320 N=320 torch.float32': {'pad_to': 273, 'ms_at_M': 0.0191, 'ms_at_pad': 0.0101}}
- `bf16_mpad_graphs_ae_at_islands`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0217, 'ms_at_pad': 0.0205}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0254, 'ms_at_pad': 0.024}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0312, 'ms_at_pad': 0.0294}} | bf16_islands: 21 fp32 Linear modules (13.1 MB) -> bf16, input cast to bf16, output cast back
- `bf16_mpad_graphs_islands`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 80 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 209, 'ms_at_M': 0.0214, 'ms_at_pad': 0.02}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 193, 'ms_at_M': 0.0266, 'ms_at_pad': 0.0241}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 241, 'ms_at_M': 0.0314, 'ms_at_pad': 0.0298}} | bf16_islands: 21 fp32 Linear modules (13.1 MB) -> bf16, input cast to bf16, output cast back
- `bf16_mpad_graphs_nofuse_islands`: SmolVLMVisionEmbeddings.forward: boolean-mask index_put replaced by the identical static position ids | no cast needed (SmolVLAPolicy has no fp32 vision island: LeRobot loads its vision tower in bf16 already); this rung equals eager_tf32 for this model | linear_mpad(lm_expert): 144 Linear modules padded; shapes padded: {'M=177 K=960 N=960 torch.bfloat16': {'pad_to': 305, 'ms_at_M': 0.0225, 'ms_at_pad': 0.0141}, 'M=177 K=960 N=2560 torch.bfloat16': {'pad_to': 209, 'ms_at_M': 0.0147, 'ms_at_pad': 0.0134}, 'M=177 K=2560 N=960 torch.bfloat16': {'pad_to': 289, 'ms_at_M': 0.0332, 'ms_at_pad': 0.0258}, 'M=50 K=720 N=320 torch.bfloat16': {'pad_to': 114, 'ms_at_M': 0.0237, 'ms_at_pad': 0.02}, 'M=50 K=720 N=2048 torch.bfloat16': {'pad_to': 194, 'ms_at_M': 0.0207, 'ms_at_pad': 0.0111}, 'M=177 K=320 N=320 torch.float32': {'pad_to': 353, 'ms_at_M': 0.0199, 'ms_at_pad': 0.0098}} | Inductor's SDPA pattern rewrite (fuse_attention._sfdp_init) disabled | bf16_islands: 21 fp32 Linear modules (13.1 MB) -> bf16, input cast to bf16, output cast back

