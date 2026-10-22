# 20261022_E4i_groot_n17_best_search — T005, groot_n17, MAXN, jetson_clocks, gate passed per rung

Median (p5–p95) ms over n = 100 timed inferences per rung, each rung in its own process; regions installed after compilation. `other` = e2e − vision − prefix − expert per iteration. GR00T e2e = `Gr00tPolicy.get_action`; `model call` = `model.get_action`, the Orin's GR00T e2e.

## N = 4

| rung | status | e2e | vision | prefix | expert | other | model call | AE % of model call | ms/step (instr.) | first call s | graphs in timed | thr. | state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ae_graphs` | measured | 142.41 (136.90–145.27) | 28.76 (28.63–29.01) | 24.90 (24.72–26.23) | 62.42 (62.27–67.98) | 26.22 (20.82–27.52) | 123.73 (123.45–130.83) | 50.4 | — | 102.93 | 0 | False | normal->normal |
| `ae_graphs_autotune` | measured | 134.52 (133.13–138.11) | 29.88 (29.68–30.21) | 25.83 (25.47–27.15) | 51.73 (51.33–53.38) | 27.03 (26.38–28.12) | 115.13 (114.32–118.53) | 44.9 | — | 178.4 | 0 | False | normal->normal |
| `ae_graphs_mpad` | measured | 145.73 (140.47–152.61) | 30.73 (30.61–31.02) | 25.56 (25.35–26.84) | 63.02 (62.87–68.15) | 26.32 (21.09–28.10) | 126.93 (126.64–133.64) | 49.6 | — | 103.86 | 0 | False | normal->normal |
| `ae_graphs_mpad_autotune` | measured | 134.96 (133.13–142.93) | 30.97 (30.78–32.76) | 25.92 (25.57–27.73) | 51.28 (50.76–52.99) | 26.79 (25.80–29.33) | 115.79 (114.96–121.64) | 44.3 | — | 182.42 | 0 | False | normal->normal |
| `all_compile_autotune_mpad` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `llm_compile_autotune` | measured | 138.98 (133.18–143.79) | 29.10 (28.85–30.57) | 31.52 (31.10–33.01) | 52.23 (51.97–52.60) | 26.40 (20.90–27.75) | 120.55 (119.65–124.20) | 43.3 | — | 222.52 | 0 | False | normal->normal |
| `llm_graphs_autotune` | measured | 138.65 (134.06–140.24) | 29.08 (28.93–30.25) | 31.70 (31.50–33.51) | 51.63 (51.48–52.23) | 26.16 (21.07–27.03) | 120.30 (119.66–123.80) | 42.9 | — | 202.97 | 0 | False | normal->normal |
| `llm_graphs_mpad_autotune` | measured | 140.13 (134.83–143.37) | 31.70 (30.82–32.36) | 34.21 (33.96–36.13) | 47.82 (47.58–48.29) | 26.20 (21.24–27.10) | 122.02 (120.15–124.37) | 39.2 | — | 207.64 | 0 | False | normal->normal |
| `vlm_compile_scalar` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `vlm_graphs_autotune` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |
| `vlm_graphs_mpad_autotune` | failed: TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function flash_attn._flash_attn_varlen_forward(*(FakeTensor(..., device='cuda:0', size=( | | | | | | | | | | | | |

## Equivalence (10 seeded-noise inferences; `equivalence.csv`)

| rung | reference | max abs Δ | relative L2 Δ | finite |
|---|---|---|---|---|
| `ae_graphs_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.6428e-02 | 5.4285e-03 | True |
| `ae_graphs_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.6428e-02 | 5.4285e-03 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.1578e-02 | 4.8368e-03 | True |
| `ae_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.1578e-02 | 4.8368e-03 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.6428e-02 | 5.4285e-03 | True |
| `ae_graphs_mpad_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.6428e-02 | 5.4285e-03 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.1578e-02 | 4.9647e-03 | True |
| `ae_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.1578e-02 | 4.9647e-03 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 3.0333e-02 | 7.1278e-03 | True |
| `llm_compile_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 3.0333e-02 | 7.1278e-03 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.8767e-02 | 7.1106e-03 | True |
| `llm_graphs_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.8767e-02 | 7.1106e-03 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_fp32_N4` | 2.9879e-02 | 8.1184e-03 | True |
| `llm_graphs_mpad_autotune_N4` | `20261022_E4d_groot_n17_ladder:eager_tf32_N4` | 2.9879e-02 | 8.1184e-03 | True |

## Preparations

- `ae_graphs_mpad`: linear_mpad(backbone+action_head): 168 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.077, 'ms_at_pad': 0.0612}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.0387, 'ms_at_pad': 0.0322}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0823, 'ms_at_pad': 0.0746}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.1006, 'ms_at_pad': 0.0704}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.056, 'ms_at_pad': 0.0477}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.0482, 'ms_at_pad': 0.0335}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1064, 'ms_at_pad': 0.0622}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 481, 'ms_at_M': 0.1915, 'ms_at_pad': 0.1673}}
- `ae_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0757, 'ms_at_pad': 0.0604}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.038, 'ms_at_pad': 0.0311}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0802, 'ms_at_pad': 0.0747}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.0998, 'ms_at_pad': 0.0698}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.177, 'ms_at_pad': 0.1677}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0554, 'ms_at_pad': 0.047}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.047, 'ms_at_pad': 0.0325}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1058, 'ms_at_pad': 0.0613}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1882, 'ms_at_pad': 0.1686}}
- `all_compile_autotune_mpad`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0764, 'ms_at_pad': 0.0604}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.0388, 'ms_at_pad': 0.0313}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0818, 'ms_at_pad': 0.0752}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.1001, 'ms_at_pad': 0.0705}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.1819, 'ms_at_pad': 0.1622}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0554, 'ms_at_pad': 0.048}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.048, 'ms_at_pad': 0.0333}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.106, 'ms_at_pad': 0.0617}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 481, 'ms_at_M': 0.222, 'ms_at_pad': 0.1747}}
- `llm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0762, 'ms_at_pad': 0.0607}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1120, 'ms_at_M': 0.0385, 'ms_at_pad': 0.0313}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0806, 'ms_at_pad': 0.0746}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.0999, 'ms_at_pad': 0.0701}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.1845, 'ms_at_pad': 0.1689}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0566, 'ms_at_pad': 0.0465}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.0472, 'ms_at_pad': 0.0314}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1047, 'ms_at_pad': 0.0606}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 497, 'ms_at_M': 0.1948, 'ms_at_pad': 0.1705}}
- `vlm_compile_scalar`: torch._dynamo.config.capture_scalar_outputs = True
- `vlm_graphs_mpad_autotune`: linear_mpad(backbone+action_head): 172 Linear modules padded; shapes padded: {'M=1024 K=1024 N=3072 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.0763, 'ms_at_pad': 0.0607}, 'M=1024 K=1024 N=1024 torch.bfloat16': {'pad_to': 1200, 'ms_at_M': 0.0383, 'ms_at_pad': 0.0311}, 'M=1024 K=1024 N=4096 torch.bfloat16': {'pad_to': 1280, 'ms_at_M': 0.081, 'ms_at_pad': 0.0749}, 'M=1024 K=4096 N=1024 torch.bfloat16': {'pad_to': 1040, 'ms_at_M': 0.1001, 'ms_at_pad': 0.07}, 'M=256 K=4096 N=4096 torch.bfloat16': {'pad_to': 272, 'ms_at_M': 0.1879, 'ms_at_pad': 0.1588}, 'M=256 K=4096 N=2048 torch.bfloat16': {'pad_to': 336, 'ms_at_M': 0.0561, 'ms_at_pad': 0.0466}, 'M=273 K=2048 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.0507, 'ms_at_pad': 0.0322}, 'M=273 K=6144 N=2048 torch.bfloat16': {'pad_to': 337, 'ms_at_M': 0.1057, 'ms_at_pad': 0.0611}, 'M=273 K=8192 N=2048 torch.bfloat16': {'pad_to': 321, 'ms_at_M': 0.2051, 'ms_at_pad': 0.1729}}

