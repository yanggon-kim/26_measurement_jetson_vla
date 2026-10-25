# 20261026_E4c_pi0_opt_ladder_N4 — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 171.09 (170.15–172.34) | — | — | — | — | — | — | 0.17 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 201.12 (199.07–209.71) | 52.56 (52.21–53.93) | 110.38 (109.76–113.93) | 35.29 (34.31–38.64) | 2.84 (2.64–3.17) | 17.5 | 8.79 | 39.95 | 0 | True | normal->normal |
| `ae_graphs` | measured | 199.42 (197.62–200.95) | 51.90 (51.57–52.14) | 112.72 (111.98–113.18) | 32.06 (31.49–33.24) | 2.57 (2.46–2.81) | 16.1 | 7.95 | 19.11 | 0 | False | normal->normal |
| `bf16_base` | not applicable: checkpoint already loads bf16 (dominant parameter dtype bfloat16), so there is no fp32 base to cast | | | | | | | | | | | |
| `bf16_both` | measured | 167.39 (166.41–170.08) | 22.52 (22.32–22.96) | 110.26 (109.51–111.25) | 31.88 (31.61–33.22) | 2.55 (2.47–2.84) | 19.0 | 7.91 | 89.36 | 0 | False | normal->normal |
| `bf16_both_graphs` | measured | 163.82 (162.98–164.71) | 21.19 (21.10–21.36) | 108.18 (107.44–108.92) | 31.87 (31.69–32.15) | 2.55 (2.51–2.67) | 19.4 | 7.90 | 48.67 | 0 | False | normal->normal |
| `bf16_mpad_graphs_ae_at_kvs` | measured | 104.96 (104.54–105.50) | 20.87 (20.79–21.00) | 60.37 (59.86–60.76) | 21.22 (21.14–21.32) | 2.54 (2.50–2.60) | 20.2 | 5.24 | 91.94 | 0 | False | normal->normal |
| `bf16_norms` | measured | 222.36 (220.55–227.54) | 55.63 (55.32–57.21) | 129.97 (128.15–132.88) | 33.32 (32.76–35.38) | 3.01 (2.83–3.61) | 15.0 | 8.27 | 20.38 | 0 | False | normal->normal |
| `bf16_vision` | measured | 201.56 (194.20–353.34) | 30.44 (29.45–63.74) | 132.50 (127.39–233.27) | 35.06 (33.40–61.54) | 3.10 (2.88–7.58) | 17.4 | 8.67 | 19.92 | 0 | False | normal->normal |
| `eager_bf16` | measured | 194.06 (192.71–194.89) | 27.43 (27.25–27.60) | 111.17 (110.97–111.35) | 52.86 (51.74–53.37) | 2.66 (2.63–2.73) | 27.2 | 13.13 | 1.02 | 0 | False | normal->normal |
| `eager_fp32` | measured | 330.71 (330.24–331.39) | 168.96 (168.85–169.07) | 109.89 (109.63–110.14) | 49.24 (48.93–49.67) | 2.60 (2.55–2.67) | 14.9 | 12.24 | 1.14 | 0 | False | normal->normal |
| `eager_tf32` | measured | 214.06 (213.58–214.64) | 51.32 (51.23–51.45) | 111.76 (111.53–112.13) | 48.31 (48.01–48.72) | 2.63 (2.59–2.71) | 22.6 | 12.01 | 1.04 | 0 | False | normal->normal |
| `vis_compile` | measured | 207.99 (207.40–208.72) | 47.27 (47.13–47.44) | 124.80 (124.31–125.46) | 33.24 (33.05–33.44) | 2.64 (2.60–2.68) | 16.0 | 8.24 | 59.63 | 0 | False | normal->normal |
| `vis_graphs` | measured | 250.55 (246.44–254.86) | 57.44 (56.28–58.35) | 151.78 (148.27–155.50) | 38.58 (37.55–39.51) | 2.99 (2.96–3.19) | 15.4 | 9.47 | 34.76 | 0 | False | normal->normal |
| `vis_static_pos` | not applicable: rung vis_static_pos applies to ['smolvla'] only | | | | | | | | | | | |
| `vlm_compile` | measured | 186.89 (186.13–188.24) | 44.03 (43.84–44.30) | 106.82 (106.37–107.63) | 33.10 (32.65–34.00) | 2.84 (2.76–2.95) | 17.7 | 8.25 | 83.8 | 0 | False | normal->normal |
| `vlm_compile_nvg` | measured | 202.98 (201.51–207.11) | 57.14 (56.89–58.26) | 108.04 (107.23–109.68) | 34.94 (33.98–36.67) | 3.05 (2.92–3.93) | 17.2 | 8.66 | 77.96 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 186.75 (185.84–187.76) | 43.31 (43.17–43.51) | 107.05 (106.40–107.66) | 33.40 (33.10–33.91) | 2.95 (2.91–3.03) | 17.9 | 8.28 | 50.67 | 0 | False | normal->normal |
| `whole_graphs` | measured | 188.18 (185.99–190.05) | — | — | — | — | — | — | 86.63 | 0 | False | normal->normal |

Per-step time by slope: `whole_graphs`: 9.02 ms/step (e2e, N = [2, 4, 6], intercept 152.72).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `whole_graphs` | measured | 207.17 (205.88–208.65) | — | — | — | — | — | — | 0.33 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N4` | `eager_fp32_N4` | 1.0044e-01 | 5.5742e-03 | True |
| `ae_compile_N4` | `eager_bf16_N4` | 1.1537e-01 | 6.3843e-03 | True |
| `ae_compile_N4` | `eager_tf32_N4` | 1.2509e-01 | 5.7663e-03 | True |
| `ae_graphs_N4` | `eager_fp32_N4` | 8.3732e-02 | 5.1736e-03 | True |
| `ae_graphs_N4` | `eager_bf16_N4` | 9.8663e-02 | 5.9839e-03 | True |
| `ae_graphs_N4` | `eager_tf32_N4` | 1.0839e-01 | 5.3649e-03 | True |
| `bf16_both_N4` | `eager_fp32_N4` | 1.5324e-02 | 3.7485e-03 | True |
| `bf16_both_N4` | `eager_bf16_N4` | 1.1715e-02 | 3.5346e-03 | True |
| `bf16_both_N4` | `eager_tf32_N4` | 2.1438e-02 | 3.7017e-03 | True |
| `bf16_both_graphs_N4` | `eager_fp32_N4` | 2.1026e-02 | 4.1952e-03 | True |
| `bf16_both_graphs_N4` | `eager_bf16_N4` | 1.7681e-02 | 3.9200e-03 | True |
| `bf16_both_graphs_N4` | `eager_tf32_N4` | 2.4785e-02 | 4.1867e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N4` | `eager_fp32_N4` | 5.9387e-02 | 4.6329e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N4` | `eager_bf16_N4` | 7.4318e-02 | 5.2310e-03 | True |
| `bf16_mpad_graphs_ae_at_kvs_N4` | `eager_tf32_N4` | 8.4042e-02 | 4.7079e-03 | True |
| `bf16_norms_N4` | `eager_fp32_N4` | 6.3036e-02 | 4.6396e-03 | True |
| `bf16_norms_N4` | `eager_bf16_N4` | 7.7967e-02 | 5.4511e-03 | True |
| `bf16_norms_N4` | `eager_tf32_N4` | 8.7691e-02 | 4.8355e-03 | True |
| `bf16_vision_N4` | `eager_fp32_N4` | 9.7083e-02 | 5.4213e-03 | True |
| `bf16_vision_N4` | `eager_bf16_N4` | 1.1201e-01 | 6.2902e-03 | True |
| `bf16_vision_N4` | `eager_tf32_N4` | 1.2174e-01 | 5.7096e-03 | True |
| `eager_bf16_N4` | `eager_fp32_N4` | 1.9847e-02 | 3.3168e-03 | True |
| `eager_bf16_N4` | `eager_bf16_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_bf16_N4` | `eager_tf32_N4` | 1.7317e-02 | 3.2284e-03 | True |
| `eager_fp32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N4` | `eager_bf16_N4` | 1.9847e-02 | 3.3178e-03 | True |
| `eager_fp32_N4` | `eager_tf32_N4` | 2.4655e-02 | 2.9774e-03 | True |
| `eager_tf32_N4` | `eager_fp32_N4` | 2.4655e-02 | 2.9771e-03 | True |
| `eager_tf32_N4` | `eager_bf16_N4` | 1.7317e-02 | 3.2290e-03 | True |
| `eager_tf32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `vis_compile_N4` | `eager_fp32_N4` | 6.1592e-02 | 4.7416e-03 | True |
| `vis_compile_N4` | `eager_bf16_N4` | 7.6523e-02 | 5.5669e-03 | True |
| `vis_compile_N4` | `eager_tf32_N4` | 8.6247e-02 | 4.9002e-03 | True |
| `vis_graphs_N4` | `eager_fp32_N4` | 6.5687e-02 | 4.8108e-03 | True |
| `vis_graphs_N4` | `eager_bf16_N4` | 8.0618e-02 | 5.6227e-03 | True |
| `vis_graphs_N4` | `eager_tf32_N4` | 9.0342e-02 | 5.0001e-03 | True |
| `vlm_compile_N4` | `eager_fp32_N4` | 7.9842e-02 | 4.6519e-03 | True |
| `vlm_compile_N4` | `eager_bf16_N4` | 9.4773e-02 | 5.4677e-03 | True |
| `vlm_compile_N4` | `eager_tf32_N4` | 1.0450e-01 | 4.9299e-03 | True |
| `vlm_compile_nvg_N4` | `eager_fp32_N4` | 8.2272e-02 | 4.7169e-03 | True |
| `vlm_compile_nvg_N4` | `eager_bf16_N4` | 9.7202e-02 | 5.5493e-03 | True |
| `vlm_compile_nvg_N4` | `eager_tf32_N4` | 1.0693e-01 | 5.0090e-03 | True |
| `vlm_graphs_N4` | `eager_fp32_N4` | 1.0302e-01 | 5.1228e-03 | True |
| `vlm_graphs_N4` | `eager_bf16_N4` | 1.1795e-01 | 5.9672e-03 | True |
| `vlm_graphs_N4` | `eager_tf32_N4` | 1.2768e-01 | 5.4667e-03 | True |
| `whole_graphs_N4` | `eager_fp32_N4` | 7.3070e-02 | 4.7787e-03 | True |
| `whole_graphs_N4` | `eager_bf16_N4` | 8.8001e-02 | 5.4909e-03 | True |
| `whole_graphs_N4` | `eager_tf32_N4` | 9.7725e-02 | 4.9439e-03 | True |

## Preparations

- `bf16_both`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_both_graphs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_mpad_graphs_ae_at_kvs`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only) | linear_mpad(lm_expert, replayed from the recorded best_bf16 table): 72 Linear modules padded; shapes padded: {'M=816 K=2048 N=256 torch.bfloat16': {'pad_to': 880, 'ms_at_M': nan, 'ms_at_pad': nan}, 'M=816 K=2048 N=16384 torch.bfloat16': {'pad_to': 1024, 'ms_at_M': nan, 'ms_at_pad': nan}} | lerobot.policies.pi0.modeling_pi0.copy.deepcopy(past_key_values) -> shallow cache copy (prefix K/V tensors shared, never written)
- `bf16_norms`: 74 layer-norm modules -> bf16 (own parameters only)
- `bf16_vision`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed
- `eager_bf16`: vision_tower + multi_modal_projector -> bf16; embed_image's fp32 cast removed; 74 layer-norm modules -> bf16 (own parameters only)

