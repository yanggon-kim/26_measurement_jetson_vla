# 20261024_E4i_groot_n16_best_search_s2 — T005, groot_n16, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 111.88 (107.58–114.02) | 21.31 (21.18–21.74) | 20.63 (20.43–20.94) | 50.13 (50.00–50.38) | 19.82 (15.48–20.98) | 97.68 (97.32–98.52) | 51.3 | — | 181.56 | 0 | False | normal->normal |
| `all_compile_ae_at` | measured | 102.60 (102.07–106.50) | 17.87 (17.75–18.30) | 14.91 (14.74–15.57) | 49.48 (49.34–50.64) | 20.38 (19.99–22.81) | 87.84 (87.55–90.16) | 56.3 | — | 207.85 | 0 | False | normal->normal |
| `all_compile_ae_at_mpad` | measured | 109.17 (108.39–112.11) | 20.56 (20.45–21.00) | 15.93 (15.79–16.39) | 52.70 (52.57–52.95) | 19.86 (19.32–21.80) | 94.32 (93.99–95.51) | 55.9 | — | 210.69 | 0 | False | normal->normal |
| `all_compile_ae_vis_at` | measured | 103.43 (102.96–104.67) | 18.31 (18.21–18.45) | 15.18 (15.04–15.41) | 50.08 (49.95–50.27) | 19.88 (19.56–20.93) | 89.13 (88.95–89.83) | 56.2 | — | 235.41 | 0 | False | normal->normal |
| `all_compile_autotune` | measured | 104.44 (100.38–107.86) | 17.09 (16.98–18.06) | 16.56 (16.45–16.84) | 50.84 (50.71–51.14) | 19.92 (15.75–21.70) | 90.16 (89.83–92.41) | 56.4 | — | 247.76 | 0 | False | normal->normal |
| `llm_compile_ae_at` | measured | 107.68 (105.22–109.17) | 21.46 (21.31–21.77) | 14.98 (14.77–15.26) | 49.14 (48.98–49.38) | 22.04 (19.90–23.34) | 91.13 (90.78–91.95) | 53.9 | — | 186.39 | 0 | False | normal->normal |
| `vlm_graphs_ae_at` | measured | 98.35 (97.89–102.07) | 17.22 (17.11–17.84) | 13.85 (13.69–14.98) | 47.26 (47.11–48.22) | 20.05 (19.67–21.60) | 83.99 (83.71–87.01) | 56.3 | — | 209.46 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.5131e-02 | True |
| `ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.5131e-02 | True |
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 3.5156e-02 | 2.1975e-02 | True |
| `all_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 3.5156e-02 | 2.1975e-02 | True |
| `all_compile_ae_at_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 5.4688e-02 | 2.9978e-02 | True |
| `all_compile_ae_at_mpad_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 5.4688e-02 | 2.9978e-02 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.5391e-02 | 1.3612e-02 | True |
| `all_compile_ae_vis_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.5391e-02 | 1.3612e-02 | True |
| `all_compile_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.7344e-02 | 1.6237e-02 | True |
| `all_compile_autotune_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.7344e-02 | 1.6237e-02 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 2.5391e-02 | 1.4661e-02 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 2.5391e-02 | 1.4661e-02 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_fp32_N4` | 3.1250e-02 | 2.2041e-02 | True |
| `vlm_graphs_ae_at_N4` | `20261022_E4d_groot_n16_ladder:eager_tf32_N4` | 3.1250e-02 | 2.2041e-02 | True |

## Preparations

- `all_compile_ae_at_mpad`: linear_mpad(backbone+action_head): 218 Linear modules padded; shapes padded: {'M=972 K=1152 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.0441, 'ms_at_pad': 0.0343}, 'M=972 K=4304 N=1152 torch.bfloat16': {'pad_to': 1052, 'ms_at_M': 0.124, 'ms_at_pad': 0.0786}, 'M=243 K=4608 N=2048 torch.bfloat16': {'pad_to': 339, 'ms_at_M': 0.0616, 'ms_at_pad': 0.0497}, 'M=243 K=2048 N=2048 torch.bfloat16': {'pad_to': 323, 'ms_at_M': 0.0363, 'ms_at_pad': 0.0325}, 'M=284 K=2048 N=2048 torch.bfloat16': {'pad_to': 332, 'ms_at_M': 0.048, 'ms_at_pad': 0.032}, 'M=284 K=6144 N=2048 torch.bfloat16': {'pad_to': 348, 'ms_at_M': 0.1055, 'ms_at_pad': 0.0613}, 'M=284 K=2048 N=151680 torch.bfloat16': {'pad_to': 364, 'ms_at_M': 4.0859, 'ms_at_pad': 3.4815}, 'M=51 K=6144 N=1536 torch.bfloat16': {'pad_to': 147, 'ms_at_M': 0.0401, 'ms_at_pad': 0.0375}}

