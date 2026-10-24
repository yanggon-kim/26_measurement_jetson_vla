# 20261024_E4i_groot_n17_best_search_s3 — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 131.87 (126.93–134.43) | 29.06 (28.91–29.39) | 25.02 (24.78–25.97) | 51.19 (51.01–52.40) | 26.50 (21.30–28.06) | 112.70 (112.16–115.10) | 45.4 | — | 203.47 | 0 | False | normal->normal |
| `all_fa_compile_ae_at` | measured | 122.17 (121.27–125.98) | 18.17 (18.07–18.67) | 28.20 (27.95–31.09) | 49.70 (49.60–49.83) | 25.97 (25.24–26.66) | 103.48 (103.12–107.28) | 48.0 | — | 227.98 | 0 | False | normal->normal |
| `vis_fa_at_ae_at` | measured | 120.82 (117.27–123.14) | 20.15 (19.39–20.33) | 26.27 (24.97–26.76) | 48.05 (47.71–48.49) | 26.56 (22.10–27.99) | 102.21 (99.58–103.28) | 47.0 | — | 245.32 | 0 | False | normal->normal |
| `vis_fa_compile_ae_at` | measured | 120.75 (115.74–124.24) | 18.28 (18.17–18.76) | 25.16 (24.94–25.95) | 51.94 (51.84–52.06) | 25.36 (20.26–27.34) | 102.75 (102.49–104.30) | 50.5 | — | 219.44 | 0 | False | normal->normal |
| `vis_fa_compile_llm_graphs_ae_at` | measured | 125.15 (120.10–126.77) | 18.05 (17.93–18.17) | 29.60 (29.41–29.87) | 52.28 (52.09–52.56) | 25.42 (20.33–26.69) | 107.40 (107.10–107.88) | 48.7 | — | 220.59 | 0 | False | normal->normal |
| `vis_fa_graphs_ae_at` | measured | 114.00 (113.69–120.28) | 18.39 (18.31–18.51) | 25.45 (25.28–25.73) | 49.76 (49.65–49.87) | 20.35 (20.13–26.36) | 101.03 (100.86–101.86) | 49.2 | — | 215.37 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.6428e-02 | 4.7523e-03 | True |
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.6428e-02 | 4.7523e-03 | True |
| `all_fa_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 9.1797e-02 | 8.7237e-03 | True |
| `all_fa_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 9.1797e-02 | 8.7237e-03 | True |
| `vis_fa_at_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 5.0781e-02 | 6.8423e-03 | True |
| `vis_fa_at_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 5.0781e-02 | 6.8423e-03 | True |
| `vis_fa_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 9.3750e-02 | 1.0622e-02 | True |
| `vis_fa_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 9.3750e-02 | 1.0622e-02 | True |
| `vis_fa_compile_llm_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 6.6406e-02 | 6.4762e-03 | True |
| `vis_fa_compile_llm_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 6.6406e-02 | 6.4762e-03 | True |
| `vis_fa_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 8.9844e-02 | 1.0211e-02 | True |
| `vis_fa_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 8.9844e-02 | 1.0211e-02 | True |

## Preparations

- `all_fa_compile_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor
- `vis_fa_at_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor
- `vis_fa_compile_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor
- `vis_fa_compile_llm_graphs_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor
- `vis_fa_graphs_ae_at`: vision flash-attn varlen: max_seqlen = total vision tokens (static upper bound) instead of a CUDA tensor

