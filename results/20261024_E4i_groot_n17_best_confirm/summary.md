# 20261024_E4i_groot_n17_best_confirm — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs_autotune` | measured | 110.98 (108.49–114.63) | 29.79 (29.07–30.52) | 26.20 (25.23–26.93) | 28.92 (28.66–29.40) | 26.30 (25.34–27.67) | 93.17 (90.70–94.38) | 31.0 | — | 72.35 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs_autotune` | measured | 133.39 (128.36–137.49) | 29.01 (28.86–29.95) | 25.28 (24.79–26.48) | 53.23 (52.79–54.66) | 25.88 (21.55–28.21) | 115.26 (114.20–118.31) | 46.2 | — | 181.17 | 0 | False | normal->normal |

Per-step time by slope: `ae_graphs_autotune`: 12.21 ms/step (ae_total, N = [2, 4, 6], intercept 4.46).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs_autotune` | measured | 160.04 (157.68–162.88) | 29.31 (28.75–30.44) | 25.87 (25.18–26.60) | 77.76 (77.50–78.07) | 26.97 (25.94–28.28) | 140.93 (139.28–143.43) | 55.2 | — | 231.88 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.1578e-02 | 5.0300e-03 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.1578e-02 | 5.0300e-03 | True |

