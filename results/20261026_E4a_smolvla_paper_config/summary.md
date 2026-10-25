# 20261026_E4a_smolvla_paper_config — T005, smolvla, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration.

## N = 10

| rung | status | e2e | vision | prefix | expert | other | AE % | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `eager_fp32` | measured | 174.66 (173.62–177.92) | 15.02 (14.86–15.24) | 14.61 (14.40–15.01) | 141.44 (140.42–144.71) | 3.60 (3.53–3.75) | 81.0 | 14.05 | 0.74 | 0 | False | normal->normal |
| `eager_tf32` | measured | 164.83 (163.50–166.81) | 14.86 (14.76–15.03) | 13.97 (13.72–14.31) | 132.39 (131.56–134.26) | 3.33 (3.24–3.51) | 80.3 | 13.14 | 0.75 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `eager_fp32_N10` | `eager_fp32_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_fp32_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 5.8766e-02 | 1.1847e-02 | True |
| `eager_fp32_N10` | `eager_tf32_N10` | 5.8766e-02 | 1.1847e-02 | True |
| `eager_tf32_N10` | `eager_fp32_N10` | 5.8766e-02 | 1.1935e-02 | True |
| `eager_tf32_N10` | `20261022_E4c_smolvla_opt_ladder:eager_bf16_N10` | 0.0000e+00 | 0.0000e+00 | True |
| `eager_tf32_N10` | `eager_tf32_N10` | 0.0000e+00 | 0.0000e+00 | True |

