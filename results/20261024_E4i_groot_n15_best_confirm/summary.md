# 20261024_E4i_groot_n15_best_confirm — T005, groot_n15, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_graphs_autotune` | measured | 105.52 (97.02–106.74) | 19.28 (19.17–19.43) | 24.58 (24.48–24.67) | 22.29 (22.19–22.40) | 39.34 (30.73–40.63) | 89.16 (88.18–89.92) | 25.0 | — | 42.17 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_graphs_autotune` | measured | 119.54 (111.57–120.75) | 19.24 (19.11–19.39) | 24.51 (24.40–24.67) | 37.27 (37.16–37.41) | 38.52 (30.56–39.74) | 103.79 (103.13–104.36) | 35.9 | — | 137.67 | 0 | False | normal->normal |

Per-step time by slope: `llm_graphs_autotune`: 7.46 ms/step (ae_total, N = [2, 4, 6], intercept 7.40).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_graphs_autotune` | measured | 135.25 (127.18–136.46) | 19.23 (19.06–19.39) | 24.56 (24.40–24.71) | 52.12 (51.96–52.27) | 39.38 (31.16–40.58) | 118.76 (118.11–119.62) | 43.9 | — | 124.84 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_fp32_N4` | 2.1454e-02 | 2.1750e-02 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n15_ladder:eager_tf32_N4` | 2.1454e-02 | 2.1750e-02 | True |

