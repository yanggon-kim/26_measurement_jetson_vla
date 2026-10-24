# 20261025_E4i_groot_n17_best_confirm — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 2

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vis_fa_at_ae_at` | measured | 100.04 (93.76–101.04) | 19.82 (19.64–20.08) | 25.89 (25.38–26.22) | 27.33 (27.12–27.47) | 26.90 (21.36–27.68) | 80.66 (79.82–81.36) | 33.9 | — | 84.16 | 0 | False | normal->normal |

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vis_fa_at_ae_at` | measured | 120.39 (115.41–121.92) | 19.22 (19.11–19.39) | 25.07 (24.85–25.32) | 50.73 (50.62–51.11) | 25.37 (20.53–26.88) | 102.44 (102.17–103.28) | 49.5 | — | 240.99 | 0 | False | normal->normal |

Per-step time by slope: `vis_fa_at_ae_at`: 11.94 ms/step (ae_total, N = [2, 4, 6], intercept 3.28).

## N = 6

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `vis_fa_at_ae_at` | measured | 146.12 (140.90–147.38) | 19.49 (19.38–19.61) | 25.25 (25.07–25.54) | 75.11 (74.93–75.24) | 26.26 (21.08–27.29) | 127.30 (127.03–127.70) | 59.0 | — | 222.96 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `vis_fa_at_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 4.1016e-02 | 6.0255e-03 | True |
| `vis_fa_at_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 4.1016e-02 | 6.0255e-03 | True |

## Preparations

- `vis_fa_at_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor

