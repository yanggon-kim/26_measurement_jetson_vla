# 20261024_E4i_groot_n16_best_confirm — T005, groot_n16, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `all_compile_ae_at` | measured | 77.28 (72.73–82.00) | 18.27 (18.11–18.69) | 15.15 (14.93–15.79) | 23.54 (23.41–24.00) | 20.33 (15.86–24.44) | 62.79 (62.41–64.41) | 37.5 | — | 81.33 | 0 | False | normal->normal |
| `all_compile_ae_vis_at` | measured | 77.46 (76.76–80.43) | 17.02 (16.85–17.40) | 14.89 (14.64–15.28) | 25.19 (25.07–25.31) | 20.28 (19.83–22.66) | 62.87 (62.51–63.60) | 40.1 | — | 77.32 | 0 | False | normal->normal |
| `vlm_graphs_ae_at` | measured | 74.77 (70.84–75.39) | 17.57 (17.46–17.67) | 13.74 (12.96–13.93) | 23.54 (23.34–23.69) | 19.92 (15.84–20.55) | 60.64 (59.21–60.94) | 38.8 | — | 16.98 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `all_compile_ae_at` | measured | 101.00 (100.27–103.46) | 18.29 (18.16–18.77) | 15.09 (14.93–15.92) | 46.77 (46.65–47.65) | 20.80 (20.27–21.85) | 86.06 (85.58–88.17) | 54.3 | — | 208.72 | 0 | False | normal->normal |
| `all_compile_ae_vis_at` | measured | 102.91 (98.52–105.79) | 17.25 (17.02–18.00) | 15.02 (14.75–15.63) | 50.15 (50.04–50.37) | 20.44 (16.31–21.91) | 88.22 (87.69–90.30) | 56.9 | — | 246.78 | 0 | False | normal->normal |
| `vlm_graphs_ae_at` | measured | 98.67 (98.12–99.55) | 17.55 (17.48–17.68) | 13.78 (13.62–14.00) | 46.97 (46.84–47.10) | 20.34 (19.96–21.22) | 84.15 (83.83–84.46) | 55.8 | — | 32.6 | 0 | False | normal->normal |

Per-step time by slope: `all_compile_ae_at`: 11.57 ms/step (ae_total, N = [2, 4, 6], intercept 0.43); `all_compile_ae_vis_at`: 12.54 ms/step (ae_total, N = [2, 4, 6], intercept 0.06); `vlm_graphs_ae_at`: 11.77 ms/step (ae_total, N = [2, 4, 6], intercept -0.05).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `all_compile_ae_at` | measured | 123.52 (119.79–125.89) | 18.21 (18.06–18.72) | 15.17 (14.93–15.60) | 69.83 (69.68–70.06) | 20.23 (16.34–21.83) | 109.13 (108.60–110.06) | 64.0 | — | 222.75 | 0 | False | normal->normal |
| `all_compile_ae_vis_at` | measured | 127.18 (123.15–128.26) | 17.03 (16.88–17.32) | 14.88 (14.67–15.08) | 75.36 (75.20–75.54) | 19.94 (15.93–20.65) | 112.98 (112.61–113.68) | 66.7 | — | 236.6 | 0 | False | normal->normal |
| `vlm_graphs_ae_at` | measured | 123.13 (119.30–125.50) | 17.74 (17.62–18.10) | 14.06 (13.81–14.46) | 70.64 (70.40–70.95) | 20.63 (16.53–22.54) | 108.29 (107.79–109.09) | 65.2 | — | 230.42 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 5.0781e-02 | 2.9477e-02 | True |
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 5.0781e-02 | 2.9477e-02 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.6692e-02 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.6692e-02 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 3.9062e-02 | 2.4256e-02 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 3.9062e-02 | 2.4256e-02 | True |

