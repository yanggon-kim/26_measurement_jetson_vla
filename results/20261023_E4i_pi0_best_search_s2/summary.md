# 20261023_E4i_pi0_best_search_s2 — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at` | measured | 148.72 (147.45–152.27) | 21.19 (20.99–21.56) | 60.86 (60.31–61.43) | 64.02 (63.15–66.57) | 2.67 (2.60–2.86) | 43.0 | 6.33 | 90.76 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_islands` | measured | 149.80 (149.06–151.94) | 20.79 (20.65–21.18) | 60.56 (60.02–61.15) | 65.97 (65.61–67.44) | 2.53 (2.45–2.79) | 44.0 | 6.53 | 84.07 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_vis_at` | measured | 168.53 (167.16–169.98) | 32.02 (31.97–32.39) | 71.05 (70.01–71.83) | 62.60 (62.17–63.99) | 2.57 (2.48–2.83) | 37.1 | 6.19 | 124.75 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_vlm_at` | measured | 156.52 (155.75–157.99) | 21.13 (21.03–21.58) | 68.91 (68.52–69.32) | 63.80 (63.31–64.92) | 2.56 (2.48–2.81) | 40.8 | 6.31 | 141.75 | 0 | False | normal->normal |
| `bf16_mpad_graphs_autotune_islands` | measured | 160.16 (159.62–161.44) | 27.75 (27.67–27.93) | 64.28 (64.09–64.55) | 65.41 (65.02–66.46) | 2.69 (2.59–2.81) | 40.8 | 6.46 | 103.65 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.4084e-02 | 3.1274e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.7558e-02 | 3.7156e-03 | True |
| `bf16_mpad_graphs_ae_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.2180e-02 | 3.0548e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 2.0136e-02 | 4.4574e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.0511e-02 | 5.0696e-03 | True |
| `bf16_mpad_graphs_ae_at_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.9633e-02 | 4.2894e-03 | True |
| `bf16_mpad_graphs_ae_vis_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 2.0552e-02 | 4.3473e-03 | True |
| `bf16_mpad_graphs_ae_vis_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.8451e-02 | 5.1181e-03 | True |
| `bf16_mpad_graphs_ae_vis_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.9502e-02 | 4.0752e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 1.9221e-02 | 3.8368e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.7119e-02 | 4.5098e-03 | True |
| `bf16_mpad_graphs_ae_vlm_at_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 1.8170e-02 | 3.6534e-03 | True |
| `bf16_mpad_graphs_autotune_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_fp32_N10` | 2.9440e-02 | 5.3227e-03 | True |
| `bf16_mpad_graphs_autotune_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 2.7770e-02 | 6.0651e-03 | True |
| `bf16_mpad_graphs_autotune_islands_N10` | `20261022_E4c_pi0_opt_ladder:eager_tf32_N10` | 2.8936e-02 | 5.2069e-03 | True |

## Preparations

- `bf16_mpad_graphs_ae_at`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0242, 'ms_at_pad': 0.0214}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7286, 'ms_at_pad': 0.5762}}
- `bf16_mpad_graphs_ae_at_islands`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0245, 'ms_at_pad': 0.022}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7007, 'ms_at_pad': 0.5761}} | bf16_islands: 5 fp32 Linear modules (13.0 MB) -> bf16, input cast to bf16, output cast back
- `bf16_mpad_graphs_ae_vis_at`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0252, 'ms_at_pad': 0.022}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7402, 'ms_at_pad': 0.5819}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 1008, 'ms_at_M': 2.0072, 'ms_at_pad': 1.4023}}
- `bf16_mpad_graphs_ae_vlm_at`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 90 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0242, 'ms_at_pad': 0.022}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.7315, 'ms_at_pad': 0.5783}, 'M=816 K=16384 N=2048 torch.bfloat16': {'pad_to': 848, 'ms_at_M': 1.7135, 'ms_at_pad': 1.2088}}
- `bf16_mpad_graphs_autotune_islands`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': 0.0245, 'ms_at_pad': 0.0213}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': 1.66, 'ms_at_pad': 0.5772}} | bf16_islands: 5 fp32 Linear modules (13.0 MB) -> bf16, input cast to bf16, output cast back

