# 20261024_E4i_groot_n17_best_confirm2 — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 107.96 (102.51–111.36) | 28.79 (28.63–29.95) | 25.53 (25.32–26.53) | 27.36 (27.15–27.93) | 26.36 (20.85–27.56) | 89.05 (88.61–91.77) | 30.7 | — | 88.86 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 133.87 (132.42–137.59) | 29.47 (29.18–29.90) | 26.07 (25.84–27.02) | 51.14 (50.91–52.62) | 27.21 (25.80–29.46) | 114.33 (113.77–116.86) | 44.7 | — | 202.74 | 0 | False | normal->normal |

Per-step time by slope: `ae_at`: 11.87 ms/step (ae_total, N = [2, 4, 6], intercept 3.62).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 157.86 (149.81–162.82) | 29.21 (28.87–30.80) | 25.67 (24.92–27.00) | 74.85 (74.63–77.05) | 27.03 (21.00–29.38) | 138.00 (135.98–142.44) | 54.2 | — | 235.61 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.2255e-02 | 4.9625e-03 | True |
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.2255e-02 | 4.9625e-03 | True |

