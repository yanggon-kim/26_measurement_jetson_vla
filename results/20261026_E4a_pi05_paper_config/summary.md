# 20261026_E4a_pi05_paper_config — T005, pi05, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `eager_fp32` | measured | 1399.18 (1397.91–1401.55) | 169.67 (169.58–169.82) | 849.99 (849.70–850.37) | 376.62 (375.48–378.88) | 2.74 (2.63–2.91) | 26.9 | 37.59 | 2.14 | 0 | False | normal->normal |
| `eager_tf32` | measured | 381.51 (380.29–382.93) | 51.95 (51.83–52.13) | 154.47 (153.68–155.24) | 172.55 (171.51–173.25) | 2.64 (2.60–2.73) | 45.2 | 17.18 | 1.17 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.6400e-03 | 2.7323e-03 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 5.7006e-04 | 2.1135e-04 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 5.7006e-04 | 2.1135e-04 | True |
| `eager_tf32_N10` | `20261022_E4c_pi05_opt_ladder:eager_bf16_N10` | 7.3837e-03 | 2.7272e-03 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |

