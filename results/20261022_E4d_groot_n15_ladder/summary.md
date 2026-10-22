# 20261022_E4d_groot_n15_ladder — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 115.82 (114.56–120.20) | 19.16 (19.07–19.34) | 31.34 (31.18–31.62) | 26.15 (25.99–26.43) | 39.16 (38.02–42.13) | 98.64 (98.20–99.33) | 26.5 | — | 64.23 | 0 | False | normal->normal |
| `ae_graphs` | measured | 114.23 (106.39–115.54) | 18.96 (18.81–19.08) | 31.15 (31.00–31.34) | 26.20 (26.06–26.38) | 37.96 (30.17–39.15) | 97.96 (97.50–98.39) | 26.8 | — | 28.49 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 94.87 (85.92–96.79) | 13.23 (13.12–13.39) | 21.12 (20.96–21.42) | 21.13 (20.99–21.23) | 39.39 (30.46–41.21) | 77.40 (76.79–77.90) | 27.3 | — | 42.69 | 0 | False | normal->normal |
| `eager_fp32` | measured | 118.14 (110.18–120.04) | 18.80 (18.72–18.96) | 30.98 (30.86–31.19) | 30.30 (30.19–30.50) | 37.98 (30.19–39.74) | 101.89 (101.47–102.48) | 29.7 | 11.21 | 0.12 | 0 | False | normal->normal |
| `eager_tf32` | measured | 118.54 (110.88–119.92) | 19.05 (18.93–19.20) | 31.35 (31.23–31.52) | 30.26 (30.16–30.40) | 37.93 (30.34–39.38) | 102.54 (102.15–103.02) | 29.5 | 11.18 | 0.12 | 0 | False | normal->normal |
| `llm_compile` | measured | 108.31 (100.22–110.59) | 19.17 (19.05–19.31) | 23.88 (23.64–24.18) | 26.18 (26.03–26.38) | 39.11 (30.98–41.22) | 91.23 (90.62–91.70) | 28.7 | — | 27.5 | 0 | False | normal->normal |
| `llm_graphs` | measured | 107.75 (99.99–109.68) | 18.94 (18.84–19.08) | 23.40 (23.19–23.67) | 26.38 (26.27–26.50) | 38.99 (31.10–40.52) | 90.50 (90.08–90.98) | 29.1 | — | 27.58 | 0 | False | normal->normal |
| `vis_compile` | measured | 111.53 (110.71–113.13) | 13.34 (13.22–13.44) | 33.35 (33.22–33.60) | 26.37 (26.20–26.59) | 38.54 (37.75–39.94) | 94.90 (94.56–95.60) | 27.8 | — | 28.3 | 0 | False | normal->normal |
| `vis_graphs` | measured | 110.55 (102.54–114.23) | 11.94 (11.85–12.04) | 33.69 (33.55–33.94) | 26.41 (26.26–26.59) | 38.43 (30.53–41.82) | 94.02 (93.52–94.56) | 28.1 | — | 29.71 | 0 | False | normal->normal |
| `vlm_compile` | measured | 101.87 (101.11–103.39) | 12.89 (12.81–12.98) | 23.78 (23.48–24.18) | 26.52 (26.38–26.74) | 38.68 (38.10–39.91) | 85.19 (84.74–85.75) | 31.1 | — | 30.83 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 102.59 (95.05–104.39) | 13.64 (13.51–13.79) | 23.47 (23.20–23.73) | 26.33 (26.23–26.47) | 39.20 (31.58–40.74) | 85.62 (85.11–86.26) | 30.8 | — | 29.31 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 135.20 (129.17–137.35) | 19.24 (19.10–19.38) | 31.31 (31.17–31.52) | 44.81 (44.61–45.11) | 39.91 (33.96–41.87) | 117.43 (117.01–117.95) | 38.2 | — | 122.99 | 0 | False | normal->normal |
| `ae_graphs` | measured | 133.02 (125.59–135.28) | 19.03 (18.90–19.14) | 31.19 (31.02–31.36) | 44.95 (44.73–45.24) | 37.82 (30.41–39.93) | 116.78 (116.31–117.28) | 38.5 | — | 57.2 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 108.44 (100.11–110.09) | 13.30 (13.18–13.46) | 21.22 (21.06–21.48) | 35.40 (35.29–35.56) | 38.54 (30.15–39.88) | 91.92 (91.51–92.60) | 38.5 | — | 193.89 | 0 | False | normal->normal |
| `eager_fp32` | measured | 141.87 (140.12–149.34) | 18.87 (18.73–19.27) | 31.04 (30.87–31.80) | 52.95 (52.71–54.38) | 38.84 (37.43–44.78) | 124.74 (124.16–128.30) | 42.4 | 11.28 | 0.85 | 0 | False | normal->normal |
| `eager_tf32` | measured | 141.23 (133.28–143.35) | 19.06 (18.91–19.22) | 31.34 (31.17–31.54) | 52.69 (52.49–52.91) | 38.02 (30.26–39.40) | 125.05 (124.67–125.71) | 42.1 | 11.20 | 0.82 | 0 | False | normal->normal |
| `llm_compile` | measured | 126.43 (118.25–127.97) | 19.23 (19.08–19.41) | 24.05 (23.71–24.35) | 44.88 (44.70–45.13) | 38.23 (30.32–39.50) | 110.03 (109.58–110.58) | 40.8 | — | 76.49 | 0 | False | normal->normal |
| `llm_graphs` | measured | 125.74 (117.75–127.32) | 19.01 (18.93–19.17) | 23.52 (23.33–23.80) | 45.34 (45.17–45.51) | 37.83 (29.85–39.42) | 109.72 (109.30–110.17) | 41.3 | — | 65.1 | 0 | False | normal->normal |
| `vis_compile` | measured | 130.73 (129.91–132.68) | 13.40 (13.28–13.53) | 33.35 (33.20–33.55) | 45.24 (44.98–45.54) | 38.76 (37.97–40.37) | 113.82 (113.36–114.58) | 39.7 | — | 99.75 | 0 | False | normal->normal |
| `vis_graphs` | measured | 129.21 (121.34–130.14) | 11.96 (11.89–12.05) | 33.68 (33.50–33.87) | 45.28 (45.08–45.49) | 38.25 (30.41–39.32) | 112.80 (112.32–113.43) | 40.1 | — | 78.1 | 0 | False | normal->normal |
| `vlm_compile` | measured | 120.23 (119.47–121.36) | 12.93 (12.82–13.02) | 23.73 (23.51–23.98) | 45.19 (45.02–45.47) | 38.37 (37.79–39.71) | 103.83 (103.42–104.33) | 43.5 | — | 109.57 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 120.75 (112.58–122.67) | 13.71 (13.61–13.83) | 23.51 (23.38–23.83) | 45.10 (44.92–45.32) | 38.34 (30.33–39.92) | 104.46 (104.07–105.14) | 43.2 | — | 99.14 | 0 | False | normal->normal |

Per-step time by slope: `ae_compile`: 9.30 ms/step (ae_total, N = [2, 4, 6], intercept 7.57); `ae_graphs`: 9.37 ms/step (ae_total, N = [2, 4, 6], intercept 7.47); `all_compile_autotune`: 7.12 ms/step (ae_total, N = [2, 4, 6], intercept 6.91); `eager_fp32`: 11.29 ms/step (ae_total, N = [2, 4, 6], intercept 7.75); `eager_tf32`: 11.27 ms/step (ae_total, N = [2, 4, 6], intercept 7.68); `llm_compile`: 9.35 ms/step (ae_total, N = [2, 4, 6], intercept 7.48); `llm_graphs`: 9.46 ms/step (ae_total, N = [2, 4, 6], intercept 7.47); `vis_compile`: 9.40 ms/step (ae_total, N = [2, 4, 6], intercept 7.59); `vis_graphs`: 9.43 ms/step (ae_total, N = [2, 4, 6], intercept 7.55); `vlm_compile`: 9.26 ms/step (ae_total, N = [2, 4, 6], intercept 8.04); `vlm_graphs`: 9.38 ms/step (ae_total, N = [2, 4, 6], intercept 7.56).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_compile` | measured | 153.87 (152.52–156.28) | 19.33 (19.21–19.53) | 31.28 (31.13–31.52) | 63.34 (63.11–64.49) | 39.86 (37.85–41.48) | 135.91 (135.55–137.49) | 46.6 | — | 166.25 | 0 | False | normal->normal |
| `ae_graphs` | measured | 152.88 (144.50–154.46) | 18.99 (18.88–19.16) | 31.16 (31.02–31.41) | 63.67 (63.40–63.85) | 39.09 (30.71–40.73) | 135.53 (135.06–136.10) | 47.0 | — | 81.94 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 123.16 (114.68–124.76) | 13.20 (13.08–13.29) | 21.07 (20.92–21.30) | 49.59 (49.45–49.78) | 39.30 (30.79–40.75) | 105.70 (105.27–106.39) | 46.9 | — | 125.04 | 0 | False | normal->normal |
| `eager_fp32` | measured | 163.35 (162.54–164.85) | 18.83 (18.74–18.98) | 30.97 (30.84–31.11) | 75.45 (75.24–75.71) | 38.11 (37.39–39.63) | 146.96 (146.67–147.36) | 51.3 | 11.25 | 0.16 | 0 | False | normal->normal |
| `eager_tf32` | measured | 163.92 (155.73–166.42) | 19.07 (18.94–19.25) | 31.36 (31.20–31.53) | 75.34 (75.12–75.57) | 38.18 (30.03–40.56) | 147.63 (147.15–148.32) | 51.0 | 11.23 | 0.16 | 0 | False | normal->normal |
| `llm_compile` | measured | 146.40 (138.21–148.22) | 19.18 (19.08–19.34) | 23.88 (23.57–24.17) | 63.59 (63.42–63.83) | 39.70 (31.56–41.31) | 128.64 (128.24–129.08) | 49.4 | — | 80.83 | 0 | False | normal->normal |
| `llm_graphs` | measured | 146.33 (137.76–147.83) | 18.97 (18.89–19.07) | 23.44 (23.21–23.76) | 64.22 (64.02–64.45) | 39.63 (31.06–41.01) | 128.59 (128.09–129.24) | 49.9 | — | 80.05 | 0 | False | normal->normal |
| `vis_compile` | measured | 149.80 (142.95–151.26) | 13.45 (13.33–13.57) | 33.45 (33.20–33.75) | 63.97 (63.81–64.20) | 38.85 (31.87–40.14) | 132.82 (132.30–133.38) | 48.2 | — | 82.89 | 0 | False | normal->normal |
| `vis_graphs` | measured | 141.54 (140.26–150.73) | 11.98 (11.89–12.04) | 33.67 (33.51–33.90) | 64.14 (63.88–64.42) | 31.66 (30.44–40.16) | 131.64 (131.12–132.19) | 48.7 | — | 91.21 | 0 | False | normal->normal |
| `vlm_compile` | measured | 138.72 (130.89–139.73) | 12.87 (12.77–12.99) | 23.80 (23.54–24.06) | 63.57 (63.40–63.79) | 38.56 (30.75–39.53) | 122.11 (121.73–122.56) | 52.1 | — | 85.52 | 0 | False | normal->normal |
| `vlm_graphs` | measured | 139.92 (131.75–142.29) | 13.71 (13.58–13.83) | 23.54 (23.31–23.72) | 63.87 (63.68–64.08) | 38.81 (30.69–40.45) | 123.30 (122.81–123.70) | 51.8 | — | 81.01 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_compile_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.0824e-01 | True |
| `ae_compile_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.0824e-01 | True |
| `ae_graphs_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.0824e-01 | True |
| `ae_graphs_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.0824e-01 | True |
| `all_compile_autotune_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.1797e-01 | True |
| `all_compile_autotune_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.1797e-01 | True |
| `eager_fp32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_fp32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N4` | `eager_tf32_N4` | 0.0000e+00 | 0.0000e+00 | True |
| `llm_compile_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.0931e-01 | True |
| `llm_compile_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.0931e-01 | True |
| `llm_graphs_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.0903e-01 | True |
| `llm_graphs_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.0903e-01 | True |
| `vis_compile_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.1976e-01 | True |
| `vis_compile_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.1976e-01 | True |
| `vis_graphs_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.1816e-01 | True |
| `vis_graphs_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.1816e-01 | True |
| `vlm_compile_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.1610e-01 | True |
| `vlm_compile_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.1610e-01 | True |
| `vlm_graphs_N4` | `eager_fp32_N4` | 1.0000e+00 | 8.1891e-01 | True |
| `vlm_graphs_N4` | `eager_tf32_N4` | 1.0000e+00 | 8.1891e-01 | True |

