# 20261022_E4d_groot_n17_ladder — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 116.13 (114.50–124.26) | 29.30 (29.07–31.57) | 25.45 (25.12–27.88) | 34.08 (33.86–36.22) | 27.00 (26.13–29.75) | 96.61 (95.87–103.65) | 35.3 | — | 70.35 | 0 | False | normal->normal |
| `ae_graphs` | measured | 113.90 (109.44–115.91) | 29.06 (28.89–29.54) | 25.24 (25.05–25.60) | 33.21 (33.12–33.37) | 26.35 (21.23–27.93) | 95.17 (94.88–95.98) | 34.9 | — | 52.39 | 0 | False | normal->normal |
| `eager_fp32` | measured | 123.02 (121.01–127.87) | 29.32 (29.02–29.97) | 26.15 (25.83–27.17) | 39.98 (39.13–40.77) | 27.12 (26.30–32.31) | 103.21 (101.77–105.33) | 38.7 | 18.22 | 0.12 | 0 | False | normal->normal |
| `eager_tf32` | measured | 117.81 (116.29–123.98) | 28.78 (28.40–29.79) | 25.08 (24.78–26.84) | 37.75 (37.63–40.72) | 26.01 (25.21–29.66) | 99.25 (98.37–104.83) | 38.0 | 17.19 | 0.12 | 0 | False | normal->normal |
| `llm_compile` | measured | 117.64 (111.82–124.10) | 29.42 (29.23–30.92) | 28.80 (28.36–32.29) | 32.92 (32.84–33.08) | 26.41 (21.02–27.77) | 98.94 (98.28–104.33) | 33.3 | — | 49.84 | 0 | False | normal->normal |
| `llm_graphs` | measured | 119.39 (113.96–124.05) | 29.05 (28.89–30.24) | 30.56 (30.31–33.11) | 33.37 (33.26–33.53) | 26.26 (21.06–27.69) | 100.76 (100.27–104.90) | 33.1 | — | 50.06 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 144.35 (138.53–148.85) | 29.12 (28.90–31.04) | 25.09 (24.85–26.58) | 63.31 (62.96–64.66) | 26.64 (21.23–28.55) | 125.26 (124.50–129.28) | 50.5 | — | 154.46 | 0 | False | normal->normal |
| `ae_graphs` | measured | 141.55 (135.83–148.20) | 28.72 (28.54–29.51) | 24.96 (24.81–26.42) | 62.28 (62.16–67.47) | 25.46 (20.09–26.55) | 123.58 (123.26–130.47) | 50.4 | — | 102.7 | 0 | False | normal->normal |
| `all_compile_autotune` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `eager_fp32` | measured | 160.99 (158.87–164.85) | 28.92 (28.68–29.62) | 25.67 (25.44–26.11) | 79.75 (78.13–81.40) | 26.51 (25.95–28.29) | 141.94 (140.07–144.46) | 56.2 | 19.06 | 0.71 | 0 | False | normal->normal |
| `eager_tf32` | measured | 152.58 (151.67–162.49) | 28.57 (28.42–29.61) | 25.07 (24.89–26.15) | 73.39 (73.14–79.32) | 25.52 (24.86–32.33) | 134.48 (134.07–143.36) | 54.6 | 17.49 | 0.65 | 0 | False | normal->normal |
| `llm_compile` | measured | 148.19 (142.36–151.38) | 29.54 (28.81–29.82) | 28.69 (28.51–29.86) | 64.19 (64.05–64.40) | 25.83 (20.47–28.18) | 130.28 (129.42–132.02) | 49.3 | — | 134.24 | 0 | False | normal->normal |
| `llm_graphs` | measured | 149.90 (142.46–152.51) | 29.43 (28.76–30.12) | 31.99 (30.25–33.07) | 63.04 (62.91–63.22) | 25.85 (20.16–27.08) | 132.40 (129.79–134.47) | 47.6 | — | 115.45 | 0 | False | normal->normal |
| `vis_compile` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `vis_graphs` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `vlm_compile` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `vlm_graphs` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |

Per-step time by slope: `ae_compile`: 15.54 ms/step (ae_total, N = [2, 4, 6], intercept 2.38); `ae_graphs`: 15.01 ms/step (ae_total, N = [2, 4, 6], intercept 2.88); `eager_fp32`: 18.74 ms/step (ae_total, N = [2, 4, 6], intercept 3.26); `eager_tf32`: 18.08 ms/step (ae_total, N = [2, 4, 6], intercept 1.43); `llm_compile`: 15.02 ms/step (ae_total, N = [2, 4, 6], intercept 3.29); `llm_graphs`: 15.27 ms/step (ae_total, N = [2, 4, 6], intercept 2.55).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 179.46 (177.98–188.46) | 29.64 (29.34–31.24) | 25.59 (25.22–27.38) | 96.24 (95.18–101.03) | 27.77 (26.85–30.20) | 159.28 (157.81–167.79) | 60.4 | — | 188.02 | 0 | False | normal->normal |
| `ae_graphs` | measured | 174.50 (168.27–177.19) | 28.99 (28.82–29.21) | 25.35 (25.10–25.70) | 93.24 (92.93–95.00) | 26.45 (21.02–28.27) | 155.56 (154.66–157.40) | 59.9 | — | 154.72 | 0 | False | normal->normal |
| `eager_fp32` | measured | 197.59 (195.56–202.33) | 29.17 (28.94–29.88) | 26.07 (25.83–26.78) | 114.95 (113.45–116.82) | 27.24 (26.51–29.03) | 178.00 (176.15–180.64) | 64.6 | 18.54 | 0.2 | 0 | False | normal->normal |
| `eager_tf32` | measured | 190.10 (189.39–200.26) | 28.67 (28.47–30.14) | 25.34 (25.09–26.97) | 110.05 (109.72–115.21) | 26.09 (25.40–28.19) | 171.55 (170.99–180.43) | 64.1 | 17.77 | 0.2 | 0 | False | normal->normal |
| `llm_compile` | measured | 177.06 (171.89–178.45) | 28.94 (28.82–29.12) | 28.43 (28.18–28.97) | 93.00 (92.46–93.22) | 26.74 (21.13–27.72) | 158.11 (157.48–158.90) | 58.8 | — | 155.82 | 0 | False | normal->normal |
| `llm_graphs` | measured | 179.70 (173.62–185.71) | 29.01 (28.86–30.28) | 30.40 (30.14–33.30) | 94.44 (92.96–94.86) | 26.41 (20.81–27.91) | 161.59 (159.81–166.48) | 58.4 | — | 155.41 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N4` | `eager_fp32_N4` | 9.6333e-01 | 2.1243e-01 | True |
| `ae_compile_N4` | `eager_tf32_N4` | 9.6333e-01 | 2.1243e-01 | True |
| `ae_graphs_N4` | `eager_fp32_N4` | 9.6946e-01 | 2.1242e-01 | True |
| `ae_graphs_N4` | `eager_tf32_N4` | 9.6946e-01 | 2.1242e-01 | True |
| `eager_fp32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `llm_compile_N4` | `eager_fp32_N4` | 9.6484e-01 | 2.1216e-01 | True |
| `llm_compile_N4` | `eager_tf32_N4` | 9.6484e-01 | 2.1216e-01 | True |
| `llm_graphs_N4` | `eager_fp32_N4` | 9.6484e-01 | 2.1200e-01 | True |
| `llm_graphs_N4` | `eager_tf32_N4` | 9.6484e-01 | 2.1200e-01 | True |

