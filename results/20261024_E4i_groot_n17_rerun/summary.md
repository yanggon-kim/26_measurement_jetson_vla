# 20261024_E4i_groot_n17_rerun — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llm_compile_ae_at` | measured | 136.50 (135.71–139.10) | 29.00 (28.84–29.17) | 28.45 (28.16–28.96) | 52.01 (51.86–52.48) | 27.13 (26.41–29.31) | 117.22 (116.85–118.46) | 44.4 | — | 211.92 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 4.3457e-02 | 8.6437e-03 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 4.3457e-02 | 8.6437e-03 | True |

