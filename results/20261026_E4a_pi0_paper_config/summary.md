# 20261026_E4a_pi0_paper_config — T005, pi0, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `eager_fp32` | measured | 407.93 (406.72–427.22) | 167.78 (167.68–167.91) | 110.88 (110.68–111.19) | 126.55 (125.42–145.77) | 2.69 (2.65–2.81) | 31.0 | 12.58 | 1.21 | 0 | False | normal->normal |
| `eager_tf32` | measured | 286.31 (284.23–287.02) | 51.59 (51.45–51.77) | 110.51 (110.30–110.73) | 121.52 (119.51–122.26) | 2.62 (2.59–2.69) | 42.4 | 12.05 | 1.11 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 9.5787e-03 | 2.5123e-03 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 8.7663e-03 | 2.0672e-03 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 8.7663e-03 | 2.0672e-03 | True |
| `eager_tf32_N10` | `20261022_E4c_pi0_opt_ladder:eager_bf16_N10` | 1.1385e-02 | 2.6709e-03 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |

