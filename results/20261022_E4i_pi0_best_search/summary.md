# 20261022_E4i_pi0_best_search — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_both_graphs_mpad` | measured | 170.86 (169.85–171.75) | 21.35 (21.20–21.56) | 64.13 (63.37–64.98) | 82.53 (82.16–83.47) | 2.67 (2.59–2.78) | 48.3 | 8.19 | 51.81 | 0 | False | normal->normal |
| `bf16_mpad_graphs` | measured | 167.14 (166.46–168.01) | 21.06 (20.97–21.19) | 63.72 (63.01–64.52) | 79.75 (79.49–80.14) | 2.59 (2.55–2.63) | 47.7 | 7.90 | 49.69 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_eager_sdpa` | measured | 170.76 (169.51–171.87) | 22.36 (22.26–22.52) | 65.15 (64.31–65.65) | 80.65 (80.07–81.48) | 2.61 (2.52–2.74) | 47.2 | 7.99 | 49.76 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_sdpa_eager` | measured | 171.86 (171.02–172.99) | 21.36 (21.27–21.50) | 67.31 (66.46–68.41) | 80.57 (80.27–81.02) | 2.63 (2.59–2.69) | 46.9 | 7.99 | 47.7 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_sdpa_sdpa` | measured | 172.54 (171.56–173.46) | 20.94 (20.85–21.07) | 68.22 (67.51–69.00) | 80.61 (80.28–81.21) | 2.66 (2.62–2.71) | 46.7 | 7.99 | 47.46 | 0 | False | normal->normal |
| `bf16_mpad_graphs_autotune` | measured | 157.83 (157.13–158.71) | 25.70 (25.62–25.79) | 66.46 (65.87–67.15) | 63.13 (62.72–63.69) | 2.51 (2.46–2.59) | 40.0 | 6.24 | 107.78 | 0 | False | normal->normal |
| `bf16_mpad_graphs_nofuse` | measured | 175.65 (174.81–176.87) | 21.21 (21.13–21.32) | 70.54 (69.89–71.41) | 81.25 (80.91–81.60) | 2.63 (2.58–2.69) | 46.2 | 8.05 | 49.99 | 0 | False | normal->normal |
| `bf16_mpad_nvg` | measured | 168.29 (167.64–169.43) | 21.85 (21.77–21.98) | 63.14 (62.30–63.93) | 80.68 (80.35–81.15) | 2.63 (2.60–2.70) | 47.9 | 7.99 | 89.82 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_fa2_eager` | measured | 171.89 (171.09–173.07) | 21.38 (21.28–21.56) | 67.46 (66.86–68.21) | 80.38 (79.98–81.01) | 2.65 (2.58–2.75) | 46.8 | 7.96 | 61.0 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_fa2_sdpa` | measured | 174.46 (173.39–175.90) | 21.64 (21.52–21.78) | 68.00 (67.33–68.65) | 82.12 (81.78–83.07) | 2.62 (2.56–2.70) | 47.1 | 8.14 | 63.1 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_sdpa_sdpa` | measured | 182.91 (181.64–183.98) | 21.62 (21.51–21.81) | 77.60 (76.46–78.69) | 81.00 (80.54–81.59) | 2.56 (2.50–2.68) | 44.3 | 8.03 | 87.81 | 0 | False | normal->normal |
| `bf16_mpad_nvg_autotune` | measured | 170.41 (169.30–171.25) | 29.89 (29.81–29.99) | 73.28 (72.52–74.02) | 64.68 (63.48–65.21) | 2.62 (2.56–2.69) | 38.0 | 6.38 | 139.75 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs` | measured | 165.89 (165.02–166.77) | — | — | — | — | — | — | 87.4 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs_autotune` | measured | 158.18 (157.66–158.55) | — | — | — | — | — | — | 152.09 | 0 | False | normal->normal |
| `eager_bf16_mpad` | measured | 245.53 (243.55–247.71) | 28.63 (28.34–28.91) | 85.25 (84.61–85.96) | 128.78 (126.98–130.81) | 2.83 (2.78–2.88) | 52.4 | 12.76 | 0.94 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.1798e-02 | 3.2270e-03 | True |
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.3011e-02 | 3.2009e-03 | True |
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.3253e-02 | 3.4188e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.0701e-02 | 2.9708e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.2135e-02 | 3.1251e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.0321e-02 | 3.0589e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 9.2965e-03 | 2.7466e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.0429e-02 | 2.8540e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.0092e-02 | 2.8705e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.1400e-02 | 2.9699e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.1609e-02 | 2.9605e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.1272e-02 | 3.0951e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.2458e-02 | 2.8715e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.1423e-02 | 2.8058e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.3057e-02 | 2.9887e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 3.5944e-02 | 4.9635e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 3.6163e-02 | 5.7206e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 3.5904e-02 | 4.7275e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.0977e-02 | 2.9013e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.0507e-02 | 3.1089e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.0856e-02 | 2.9844e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.0776e-02 | 2.9670e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.0730e-02 | 2.8290e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.0393e-02 | 2.9898e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.2867e-02 | 3.2791e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.3149e-02 | 3.4232e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.2812e-02 | 3.3342e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 9.8851e-03 | 2.8671e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.1172e-02 | 3.0952e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.0835e-02 | 2.9433e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 2.0657e-02 | 3.6153e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.6054e-02 | 4.4762e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 2.0153e-02 | 3.6900e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.4680e-02 | 3.3594e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.8237e-02 | 3.7793e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.2439e-02 | 3.1509e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.0018e-02 | 2.8755e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.2295e-02 | 3.1747e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.1958e-02 | 3.0037e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 2.2145e-02 | 4.8169e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.9587e-02 | 5.6794e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 2.1642e-02 | 4.5627e-03 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 9.5787e-03 | 2.5115e-03 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.1385e-02 | 2.6700e-03 | True |

## Preparations

- `bf16_both_graphs_mpad`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 108 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0247, 'ms_at_pad': 0.0212}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.6502, 'ms_at_pad': 0.5815}, 'M=51 K=1024 N=256 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.0213, 'ms_at_pad': 0.0202}}
- `bf16_mpad_graphs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0255, 'ms_at_pad': 0.0219}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7055, 'ms_at_pad': 0.5819}}
- `bf16_mpad_graphs_attn_eager_sdpa`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 0.0254, 'ms_at_pad': 0.0239}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7013, 'ms_at_pad': 0.5781}} | expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_attn_sdpa_eager`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0213}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7108, 'ms_at_pad': 0.5774}} | prefix: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_attn_sdpa_sdpa`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0216}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.755, 'ms_at_pad': 0.5813}} | prefix: 18 attention modules -> thor_sdpa; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_autotune`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0244, 'ms_at_pad': 0.0214}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7032, 'ms_at_pad': 0.5837}}
- `bf16_mpad_graphs_nofuse`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0246, 'ms_at_pad': 0.0212}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7254, 'ms_at_pad': 0.5862}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 1056, 'ms_at_M': 1.6689, 'ms_at_pad': 1.4404}} | Inductor's SDPA pattern rewrite (fuse_attention._sfdp_init) disabled
- `bf16_mpad_nvg`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0242, 'ms_at_pad': 0.021}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.6961, 'ms_at_pad': 0.5813}}
- `bf16_mpad_nvg_attn_fa2_eager`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0212}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7157, 'ms_at_pad': 0.5787}} | prefix: 18 attention modules -> thor_fa2
- `bf16_mpad_nvg_attn_fa2_sdpa`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0251, 'ms_at_pad': 0.0217}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.6627, 'ms_at_pad': 0.5785}} | prefix: 18 attention modules -> thor_fa2; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_nvg_attn_sdpa_sdpa`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0246, 'ms_at_pad': 0.021}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1008, 'ms_at_M': 1.6874, 'ms_at_pad': 0.5868}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 1.7403, 'ms_at_pad': 1.3343}} | prefix: 18 attention modules -> thor_sdpa; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_nvg_autotune`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 992, 'ms_at_M': 0.0255, 'ms_at_pad': 0.0222}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7405, 'ms_at_pad': 0.5796}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 976, 'ms_at_M': 1.6179, 'ms_at_pad': 1.4104}}
- `bf16_mpad_whole_graphs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0252, 'ms_at_pad': 0.0213}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.6706, 'ms_at_pad': 0.5755}}
- `bf16_mpad_whole_graphs_autotune`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0248, 'ms_at_pad': 0.0217}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1008, 'ms_at_M': 1.6785, 'ms_at_pad': 0.5865}}
- `eager_bf16_mpad`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0246, 'ms_at_pad': 0.0213}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.6345, 'ms_at_pad': 0.5803}}

