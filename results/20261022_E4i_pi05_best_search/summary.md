# 20261022_E4i_pi05_best_search — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_both_graphs_mpad` | measured | 183.87 (183.20–184.77) | 21.34 (21.23–21.54) | 66.15 (65.75–66.62) | 93.59 (93.15–94.24) | 2.75 (2.70–2.85) | 50.9 | 9.29 | 49.2 | 0 | False | normal->normal |
| `bf16_mpad_graphs` | measured | 184.05 (183.16–185.11) | 21.06 (20.94–21.28) | 68.41 (67.81–69.26) | 91.85 (91.55–92.66) | 2.61 (2.55–2.79) | 49.9 | 9.12 | 50.9 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_eager_sdpa` | measured | 204.45 (203.71–205.15) | 21.97 (21.83–22.13) | 79.51 (79.26–79.89) | 100.24 (99.77–100.83) | 2.65 (2.57–2.78) | 49.0 | 9.95 | 48.3 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_sdpa_eager` | measured | 191.85 (190.98–192.85) | 21.56 (21.45–21.69) | 74.00 (73.26–74.61) | 93.50 (93.09–94.15) | 2.73 (2.69–2.82) | 48.7 | 9.28 | 48.51 | 0 | False | normal->normal |
| `bf16_mpad_graphs_attn_sdpa_sdpa` | measured | 200.27 (199.30–201.11) | 21.98 (21.87–22.19) | 72.80 (72.19–73.53) | 102.77 (102.33–103.27) | 2.61 (2.56–2.72) | 51.3 | 10.21 | 47.2 | 0 | False | normal->normal |
| `bf16_mpad_graphs_autotune` | measured | 189.34 (188.55–190.31) | 31.33 (31.25–31.46) | 70.41 (69.79–71.09) | 84.95 (84.60–85.37) | 2.65 (2.57–2.76) | 44.9 | 8.42 | 106.46 | 0 | False | normal->normal |
| `bf16_mpad_graphs_nofuse` | measured | 185.14 (184.23–186.23) | 21.30 (21.19–21.46) | 67.99 (67.35–69.01) | 93.11 (92.60–93.54) | 2.63 (2.56–2.72) | 50.3 | 9.23 | 49.88 | 0 | False | normal->normal |
| `bf16_mpad_nvg` | measured | 184.76 (183.92–185.61) | 20.89 (20.76–21.09) | 68.43 (67.72–68.92) | 92.76 (92.32–93.21) | 2.70 (2.63–2.77) | 50.2 | 9.20 | 80.18 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_fa2_eager` | measured | 191.19 (189.30–192.89) | 21.35 (21.21–21.54) | 72.76 (71.92–73.81) | 94.24 (92.74–95.09) | 2.80 (2.74–2.94) | 49.3 | 9.34 | 60.3 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_fa2_fa2` | measured | 271.35 (267.73–289.77) | 21.77 (21.61–21.97) | 73.20 (72.39–74.25) | 173.15 (170.26–191.23) | 3.11 (3.01–3.25) | 63.8 | 17.25 | 57.69 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_fa2_sdpa` | measured | 198.96 (197.84–200.46) | 21.81 (21.67–21.97) | 71.10 (70.29–72.04) | 103.29 (102.76–104.41) | 2.74 (2.65–2.87) | 51.9 | 10.26 | 61.89 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_sdpa_fa2` | measured | 268.12 (266.08–271.68) | 21.35 (21.22–21.48) | 77.21 (76.61–78.01) | 166.26 (164.46–169.38) | 3.07 (3.01–3.13) | 62.0 | 16.51 | 84.11 | 0 | False | normal->normal |
| `bf16_mpad_nvg_attn_sdpa_sdpa` | measured | 201.31 (200.36–202.24) | 21.85 (21.74–22.00) | 74.47 (73.92–75.06) | 102.31 (101.95–102.77) | 2.66 (2.60–2.75) | 50.8 | 10.16 | 73.94 | 0 | False | normal->normal |
| `bf16_mpad_nvg_autotune` | measured | 190.99 (190.17–193.53) | 33.12 (33.06–33.31) | 71.18 (70.51–72.02) | 83.94 (83.55–85.80) | 2.65 (2.58–2.85) | 44.0 | 8.33 | 133.52 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs` | measured | 186.21 (185.32–187.02) | — | — | — | — | — | — | 85.3 | 0 | False | normal->normal |
| `bf16_mpad_whole_graphs_autotune` | measured | 187.93 (186.33–189.32) | — | — | — | — | — | — | 157.67 | 0 | False | normal->normal |
| `eager_bf16_mpad` | measured | 289.39 (287.29–291.90) | 29.06 (28.91–29.19) | 97.69 (96.98–98.19) | 159.84 (157.95–162.19) | 2.85 (2.81–2.92) | 55.2 | 15.88 | 0.99 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 3.8551e-03 | 1.8390e-03 | True |
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.5751e-03 | 2.7490e-03 | True |
| `bf16_both_graphs_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 3.5914e-03 | 1.8529e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.2580e-03 | 1.8753e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.4394e-03 | 2.7488e-03 | True |
| `bf16_mpad_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 3.8416e-03 | 1.8782e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.1458e-03 | 1.9001e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.9414e-03 | 2.7559e-03 | True |
| `bf16_mpad_graphs_attn_eager_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 3.7812e-03 | 1.9038e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.2659e-03 | 1.8877e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.0549e-03 | 2.7788e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 4.5269e-03 | 1.8988e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.7062e-03 | 1.6829e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.3095e-03 | 2.6986e-03 | True |
| `bf16_mpad_graphs_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 4.3812e-03 | 1.6763e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.4944e-03 | 2.0511e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.3961e-03 | 2.9954e-03 | True |
| `bf16_mpad_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.1694e-03 | 2.0341e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.1667e-03 | 1.8205e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.2711e-03 | 2.7874e-03 | True |
| `bf16_mpad_graphs_nofuse_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 4.7503e-03 | 1.8089e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.9366e-03 | 2.0157e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.9628e-03 | 2.5911e-03 | True |
| `bf16_mpad_nvg_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.5203e-03 | 1.9946e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.3983e-03 | 2.3712e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.9577e-03 | 2.8927e-03 | True |
| `bf16_mpad_nvg_attn_fa2_eager_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.1899e-03 | 2.3556e-03 | True |
| `bf16_mpad_nvg_attn_fa2_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 7.9490e-03 | 2.0338e-03 | True |
| `bf16_mpad_nvg_attn_fa2_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.8904e-03 | 2.8953e-03 | True |
| `bf16_mpad_nvg_attn_fa2_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 8.2100e-03 | 2.0718e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 8.4097e-03 | 2.2794e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 1.1129e-02 | 3.0913e-03 | True |
| `bf16_mpad_nvg_attn_fa2_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 8.9419e-03 | 2.3079e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 6.9058e-03 | 2.0407e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 8.5988e-03 | 2.8633e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_fa2_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 6.5808e-03 | 2.0251e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.9126e-03 | 2.0607e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.4432e-03 | 2.7381e-03 | True |
| `bf16_mpad_nvg_attn_sdpa_sdpa_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.4962e-03 | 2.0473e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 6.3862e-03 | 2.1956e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 1.0811e-02 | 3.0331e-03 | True |
| `bf16_mpad_nvg_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 6.6472e-03 | 2.2001e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 5.0569e-03 | 2.2826e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.5560e-03 | 2.7291e-03 | True |
| `bf16_mpad_whole_graphs_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 4.7319e-03 | 2.2744e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 4.7645e-03 | 2.6720e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 9.6793e-03 | 3.0878e-03 | True |
| `bf16_mpad_whole_graphs_autotune_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 5.0255e-03 | 2.6592e-03 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_fp32_N10` | 7.6400e-03 | 2.7309e-03 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_mpad_N10` | `20261022_E4c_pi05_opt_ladder:eager_tf32_N10` | 7.3837e-03 | 2.7257e-03 | True |

## Preparations

- `bf16_both_graphs_mpad`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0879, 'ms_at_pad': 0.0789}}
- `bf16_mpad_graphs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0879, 'ms_at_pad': 0.0781}}
- `bf16_mpad_graphs_attn_eager_sdpa`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 54 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0864, 'ms_at_pad': 0.0783}, 'M=968 K=16384 N=2048 torch.bfloat16': {'pad_to': 1064, 'ms_at_M': 1.6316, 'ms_at_pad': 1.5315}} | expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_attn_sdpa_eager`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0869, 'ms_at_pad': 0.0776}} | prefix: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_attn_sdpa_sdpa`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0873, 'ms_at_pad': 0.0787}} | prefix: 18 attention modules -> thor_sdpa; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_graphs_autotune`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0855, 'ms_at_pad': 0.0781}}
- `bf16_mpad_graphs_nofuse`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0866, 'ms_at_pad': 0.0779}} | Inductor's SDPA pattern rewrite (fuse_attention._sfdp_init) disabled
- `bf16_mpad_nvg`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1128, 'ms_at_M': 0.0872, 'ms_at_pad': 0.0778}}
- `bf16_mpad_nvg_attn_fa2_eager`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0865, 'ms_at_pad': 0.0789}} | prefix: 18 attention modules -> thor_fa2
- `bf16_mpad_nvg_attn_fa2_fa2`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1128, 'ms_at_M': 0.0868, 'ms_at_pad': 0.0787}} | prefix: 18 attention modules -> thor_fa2; expert: 18 attention modules -> thor_fa2
- `bf16_mpad_nvg_attn_fa2_sdpa`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0871, 'ms_at_pad': 0.0789}} | prefix: 18 attention modules -> thor_fa2; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_nvg_attn_sdpa_fa2`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1128, 'ms_at_M': 0.0864, 'ms_at_pad': 0.0784}} | prefix: 18 attention modules -> thor_sdpa; expert: 18 attention modules -> thor_fa2
- `bf16_mpad_nvg_attn_sdpa_sdpa`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0866, 'ms_at_pad': 0.0791}} | prefix: 18 attention modules -> thor_sdpa; expert: 18 attention modules -> thor_sdpa
- `bf16_mpad_nvg_autotune`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0874, 'ms_at_pad': 0.0786}}
- `bf16_mpad_whole_graphs`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0861, 'ms_at_pad': 0.0784}}
- `bf16_mpad_whole_graphs_autotune`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0856, 'ms_at_pad': 0.0782}, 'M=50 K=1024 N=256 torch.bfloat16': {'pad_to': 290, 'ms_at_M': 0.0223, 'ms_at_pad': 0.021}}
- `eager_bf16_mpad`: paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') (LeRobot's keep-list stays fp32); vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 37 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 36 Linear modules padded; shapes padded: {'M=968 K=2048 N=2048 torch.bfloat16': {'pad_to': 1144, 'ms_at_M': 0.0871, 'ms_at_pad': 0.0784}}

