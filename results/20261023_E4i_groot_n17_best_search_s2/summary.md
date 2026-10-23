# 20261023_E4i_groot_n17_best_search_s2 — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 140.23 (133.18–362.82) | 30.31 (28.88–103.64) | 26.70 (24.96–108.78) | 52.97 (51.30–94.14) | 29.97 (27.81–60.78) | 118.67 (112.81–336.72) | 44.6 | — | 238.37 | 0 | False | normal->normal |
| `ae_at_mpad` | measured | 141.98 (135.86–371.31) | 32.55 (31.64–101.74) | 27.81 (26.38–114.34) | 51.32 (50.62–87.35) | 29.35 (26.78–67.94) | 120.36 (116.98–340.74) | 42.6 | — | 221.33 | 0 | False | normal->normal |
| `llm_compile_ae_at` | measured | 296.98 (136.41–503.39) | 80.58 (29.41–158.83) | 79.34 (29.20–143.80) | 71.74 (47.98–115.12) | 70.94 (28.00–100.97) | 275.72 (115.57–480.16) | 26.0 | — | 233.66 | 0 | False | normal->normal |
| `llm_graphs_ae_at` | measured | 157.23 (150.12–164.18) | 30.34 (29.01–31.56) | 34.10 (31.03–35.96) | 60.98 (60.04–63.09) | 30.96 (29.50–35.19) | 134.18 (128.19–137.03) | 45.5 | — | 246.7 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.1578e-02 | 4.6683e-03 | True |
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.1578e-02 | 4.6683e-03 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.5037e-02 | 4.6153e-03 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.5037e-02 | 4.6153e-03 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 3.5889e-02 | 7.9246e-03 | True |
| `llm_compile_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 3.5889e-02 | 7.9246e-03 | True |
| `llm_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 4.2466e-02 | 6.9866e-03 | True |
| `llm_graphs_ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 4.2466e-02 | 6.9866e-03 | True |

## Preparations

- `ae_at_mpad`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0767, 'ms_at_pad': 0.0605}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1200, 'ms_at_M': 0.0382, 'ms_at_pad': 0.0316}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1216, 'ms_at_M': 0.0808, 'ms_at_pad': 0.075}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.0998, 'ms_at_pad': 0.07}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.1739, 'ms_at_pad': 0.1616}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0552, 'ms_at_pad': 0.0469}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.0479, 'ms_at_pad': 0.0321}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1056, 'ms_at_pad': 0.061}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1848, 'ms_at_pad': 0.164}}

