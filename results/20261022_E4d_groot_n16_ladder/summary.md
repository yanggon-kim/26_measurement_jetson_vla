# 20261022_E4d_groot_n16_ladder — T005, groot_n16, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 95.19 (91.19–96.45) | 21.34 (21.13–21.58) | 20.63 (20.38–20.95) | 33.02 (32.91–33.18) | 20.13 (16.15–20.84) | 80.54 (80.12–81.08) | 41.0 | — | 87.54 | 0 | False | normal->normal |
| `ae_graphs` | measured | 93.97 (89.72–94.97) | 21.28 (21.16–21.48) | 20.59 (20.37–20.79) | 32.45 (32.30–32.55) | 19.70 (15.52–20.39) | 79.87 (79.52–80.37) | 40.6 | — | 49.25 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 77.31 (72.97–78.25) | 17.05 (16.92–17.20) | 16.36 (16.26–16.49) | 24.31 (24.20–24.42) | 19.58 (15.38–20.36) | 63.35 (63.17–63.57) | 38.4 | — | 69.36 | 0 | False | normal->normal |
| `eager_fp32` | measured | 98.79 (98.13–100.62) | 21.01 (20.89–21.22) | 20.53 (20.36–20.91) | 36.73 (36.52–37.07) | 20.35 (19.96–21.57) | 83.83 (83.52–84.62) | 43.8 | 18.29 | 0.1 | 0 | False | normal->normal |
| `eager_tf32` | measured | 98.94 (98.33–101.32) | 21.22 (21.08–21.60) | 20.67 (20.52–21.18) | 36.92 (36.73–37.21) | 20.01 (19.65–21.07) | 84.28 (83.98–85.28) | 43.8 | 18.38 | 0.1 | 0 | False | normal->normal |
| `llm_compile` | measured | 89.18 (87.69–92.63) | 21.42 (21.26–22.23) | 15.09 (14.88–16.03) | 32.21 (32.09–33.15) | 20.25 (19.46–22.42) | 74.51 (73.98–76.93) | 43.2 | — | 48.75 | 0 | False | normal->normal |
| `llm_graphs` | measured | 86.76 (86.11–88.37) | 21.15 (21.04–21.34) | 13.92 (13.75–14.20) | 31.93 (31.81–32.07) | 19.76 (19.39–20.55) | 72.62 (72.36–73.25) | 44.0 | — | 47.93 | 0 | False | normal->normal |
| `vis_compile` | measured | 90.26 (87.27–91.54) | 17.54 (17.39–17.93) | 20.55 (20.35–21.29) | 32.09 (31.99–32.22) | 20.04 (16.32–20.90) | 75.85 (75.54–77.10) | 42.3 | — | 47.58 | 0 | False | normal->normal |
| `vlm_compile` | measured | 85.20 (81.35–85.83) | 18.85 (18.76–18.98) | 14.64 (14.48–14.80) | 31.90 (31.74–32.01) | 19.75 (16.04–20.43) | 70.93 (70.70–71.27) | 45.0 | — | 49.82 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 85.54 (81.25–86.85) | 18.99 (18.87–19.26) | 14.50 (14.33–14.81) | 32.17 (32.04–32.33) | 19.99 (15.70–20.82) | 71.40 (71.10–72.20) | 45.1 | — | 49.7 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 128.07 (127.32–129.30) | 21.26 (21.12–21.46) | 20.58 (20.39–20.80) | 66.01 (65.83–66.23) | 20.16 (19.60–21.03) | 113.42 (113.06–113.98) | 58.2 | — | 194.49 | 0 | False | normal->normal |
| `ae_graphs` | measured | 126.23 (121.56–128.17) | 21.14 (21.01–21.39) | 20.53 (20.36–20.75) | 64.81 (64.64–65.13) | 19.65 (15.23–20.73) | 112.13 (111.78–112.56) | 57.8 | — | 101.11 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 101.87 (97.78–102.91) | 17.09 (16.97–17.23) | 16.46 (16.32–16.62) | 48.53 (48.37–48.72) | 19.80 (15.75–20.68) | 87.73 (87.42–88.05) | 55.3 | — | 260.09 | 0 | False | normal->normal |
| `eager_fp32` | measured | 135.36 (134.70–138.93) | 21.08 (20.93–21.42) | 20.60 (20.47–21.04) | 73.46 (73.24–75.04) | 20.03 (19.74–21.32) | 120.75 (120.37–123.14) | 60.8 | 18.32 | 0.82 | 0 | False | normal->normal |
| `eager_tf32` | measured | 136.09 (135.13–138.96) | 21.26 (21.11–21.71) | 20.72 (20.52–21.21) | 73.91 (73.66–75.53) | 20.02 (19.54–21.35) | 121.48 (120.98–124.47) | 60.8 | 18.44 | 0.79 | 0 | False | normal->normal |
| `llm_compile` | measured | 121.72 (120.19–129.57) | 21.36 (21.13–22.62) | 15.21 (14.94–17.05) | 64.32 (63.96–66.54) | 20.66 (19.81–24.25) | 106.53 (105.83–111.79) | 60.4 | — | 126.94 | 0 | False | normal->normal |
| `llm_graphs` | measured | 118.63 (114.16–119.46) | 21.24 (21.09–21.46) | 13.90 (13.75–14.16) | 63.69 (63.51–63.94) | 19.78 (15.44–20.45) | 104.45 (104.17–105.08) | 61.0 | — | 106.4 | 0 | False | normal->normal |
| `vis_compile` | measured | 122.00 (117.91–122.88) | 17.56 (17.43–17.72) | 20.50 (20.33–20.70) | 63.98 (63.73–64.24) | 19.99 (15.95–20.83) | 107.73 (107.37–108.10) | 59.4 | — | 145.63 | 0 | False | normal->normal |
| `vis_graphs` | failed: RuntimeError: These storage data ptrs are not allocated in pool (0, 1) but should be {5791929663488} | | | | | | | | | | | | |
| `vlm_compile` | measured | 116.89 (112.40–119.00) | 18.85 (18.75–19.04) | 14.68 (14.48–15.13) | 63.59 (63.41–63.93) | 19.81 (15.33–20.74) | 102.77 (102.42–103.56) | 61.9 | — | 176.89 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 117.53 (113.33–118.37) | 18.90 (18.81–19.02) | 14.47 (14.31–14.66) | 64.17 (63.99–64.42) | 19.93 (15.70–20.69) | 103.27 (102.98–103.61) | 62.1 | — | 148.91 | 0 | False | normal->normal |

Per-step time by slope: `ae_compile`: 16.37 ms/step (ae_total, N = [2, 4, 6], intercept 0.35); `ae_graphs`: 16.12 ms/step (ae_total, N = [2, 4, 6], intercept 0.26); `all_compile_autotune`: 12.06 ms/step (ae_total, N = [2, 4, 6], intercept 0.24); `eager_fp32`: 18.34 ms/step (ae_total, N = [2, 4, 6], intercept 0.07); `eager_tf32`: 18.37 ms/step (ae_total, N = [2, 4, 6], intercept 0.25); `llm_compile`: 15.92 ms/step (ae_total, N = [2, 4, 6], intercept 0.47); `llm_graphs`: 15.88 ms/step (ae_total, N = [2, 4, 6], intercept 0.17); `vis_compile`: 15.90 ms/step (ae_total, N = [2, 4, 6], intercept 0.31); `vlm_compile`: 15.81 ms/step (ae_total, N = [2, 4, 6], intercept 0.31); `vlm_graphs`: 16.01 ms/step (ae_total, N = [2, 4, 6], intercept 0.15).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 160.87 (160.20–162.03) | 21.33 (21.14–21.52) | 20.62 (20.50–20.92) | 98.52 (98.31–98.81) | 20.34 (19.86–21.22) | 146.07 (145.72–146.70) | 67.5 | — | 302.35 | 0 | False | normal->normal |
| `ae_graphs` | measured | 158.57 (154.37–159.42) | 21.31 (21.20–21.43) | 20.47 (20.30–20.72) | 96.91 (96.68–97.16) | 19.95 (15.79–20.75) | 144.34 (144.03–144.75) | 67.1 | — | 158.68 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 125.71 (121.36–126.98) | 17.12 (17.00–17.56) | 16.44 (16.33–16.68) | 72.54 (72.33–72.78) | 19.61 (15.32–20.42) | 111.74 (111.39–112.40) | 64.9 | — | 227.14 | 0 | False | normal->normal |
| `eager_fp32` | measured | 172.73 (171.24–175.80) | 21.11 (20.97–21.25) | 20.55 (20.40–20.73) | 110.09 (109.40–112.80) | 20.74 (19.92–22.15) | 157.43 (156.58–160.33) | 69.9 | 18.32 | 0.17 | 0 | False | normal->normal |
| `eager_tf32` | measured | 173.18 (172.52–175.47) | 21.41 (21.18–21.60) | 20.73 (20.55–21.00) | 110.42 (110.09–112.15) | 20.56 (20.19–21.12) | 158.20 (157.74–159.67) | 69.8 | 18.37 | 0.17 | 0 | False | normal->normal |
| `llm_compile` | measured | 152.53 (151.54–154.55) | 21.40 (21.22–21.77) | 14.94 (14.72–15.63) | 95.87 (95.38–96.69) | 20.20 (19.73–21.03) | 137.89 (137.30–139.36) | 69.5 | — | 150.45 | 0 | False | normal->normal |
| `llm_graphs` | measured | 150.84 (146.88–151.84) | 21.34 (21.24–21.53) | 13.99 (13.84–14.15) | 95.45 (94.87–96.08) | 20.14 (16.11–20.74) | 136.45 (135.74–137.28) | 70.0 | — | 150.02 | 0 | False | normal->normal |
| `vis_compile` | measured | 153.90 (150.43–154.98) | 17.54 (17.40–17.69) | 20.47 (20.32–20.75) | 95.71 (95.49–95.99) | 20.19 (16.86–21.08) | 139.46 (139.09–139.91) | 68.6 | — | 154.33 | 0 | False | normal->normal |
| `vlm_compile` | measured | 148.54 (147.70–149.40) | 18.86 (18.73–19.00) | 14.64 (14.52–14.85) | 95.14 (94.89–95.43) | 19.83 (19.11–20.68) | 134.30 (133.91–134.64) | 70.8 | — | 154.46 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 149.71 (145.54–150.95) | 18.98 (18.88–19.13) | 14.49 (14.33–14.75) | 96.20 (95.96–96.53) | 20.02 (15.78–21.05) | 135.35 (135.03–135.95) | 71.1 | — | 149.98 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N4` | `eager_fp32_N4` | 7.0410e-01 | 4.0143e-01 | True |
| `ae_compile_N4` | `eager_tf32_N4` | 7.0410e-01 | 4.0143e-01 | True |
| `ae_graphs_N4` | `eager_fp32_N4` | 7.0410e-01 | 4.0124e-01 | True |
| `ae_graphs_N4` | `eager_tf32_N4` | 7.0410e-01 | 4.0124e-01 | True |
| `all_compile_autotune_N4` | `eager_fp32_N4` | 7.0020e-01 | 4.0141e-01 | True |
| `all_compile_autotune_N4` | `eager_tf32_N4` | 7.0020e-01 | 4.0141e-01 | True |
| `eager_fp32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `llm_compile_N4` | `eager_fp32_N4` | 7.0410e-01 | 4.0174e-01 | True |
| `llm_compile_N4` | `eager_tf32_N4` | 7.0410e-01 | 4.0174e-01 | True |
| `llm_graphs_N4` | `eager_fp32_N4` | 7.0410e-01 | 4.0270e-01 | True |
| `llm_graphs_N4` | `eager_tf32_N4` | 7.0410e-01 | 4.0270e-01 | True |
| `vis_compile_N4` | `eager_fp32_N4` | 7.0215e-01 | 4.0091e-01 | True |
| `vis_compile_N4` | `eager_tf32_N4` | 7.0215e-01 | 4.0091e-01 | True |
| `vlm_compile_N4` | `eager_fp32_N4` | 7.0020e-01 | 4.0257e-01 | True |
| `vlm_compile_N4` | `eager_tf32_N4` | 7.0020e-01 | 4.0257e-01 | True |
| `vlm_graphs_N4` | `eager_fp32_N4` | 7.1191e-01 | 4.0646e-01 | True |
| `vlm_graphs_N4` | `eager_tf32_N4` | 7.1191e-01 | 4.0646e-01 | True |

