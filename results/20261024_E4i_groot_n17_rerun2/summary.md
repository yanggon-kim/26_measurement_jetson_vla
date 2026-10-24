# 20261024_E4i_groot_n17_rerun2 — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_at` | measured | 127.45 (122.25–129.12) | 28.84 (28.73–28.99) | 25.16 (24.98–25.44) | 47.44 (47.25–47.73) | 26.05 (20.81–27.28) | 108.86 (108.48–109.41) | 43.6 | — | 204.05 | 0 | False | normal->normal |
| `ae_at_mpad` | measured | 134.86 (129.65–138.76) | 30.56 (30.42–30.83) | 25.67 (25.48–26.52) | 53.64 (53.50–55.74) | 25.17 (20.05–25.96) | 117.33 (117.03–120.78) | 45.7 | — | 204.19 | 0 | False | normal->normal |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.2255e-02 | 4.7982e-03 | True |
| `ae_at_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.2255e-02 | 4.7982e-03 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.1578e-02 | 4.9992e-03 | True |
| `ae_at_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.1578e-02 | 4.9992e-03 | True |

## Preparations

- `ae_at_mpad`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0765, 'ms_at_pad': 0.0603}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.038, 'ms_at_pad': 0.0309}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0804, 'ms_at_pad': 0.0745}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.1, 'ms_at_pad': 0.0693}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.1835, 'ms_at_pad': 0.1714}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0551, 'ms_at_pad': 0.0471}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.0465, 'ms_at_pad': 0.0318}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.1054, 'ms_at_pad': 0.061}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 369, 'ms_at_M': 0.1998, 'ms_at_pad': 0.1857}}

