# GR00T N1.5: where do the rejected compiled candidates diverge from eager? (T005 E4i diagnostic) — 2026-10-23

**Question.** In the stage-1 search (`results/20261022_E4i_groot_n15_best_search/`), two candidates differ from
`eager_fp32` by relative L2 0.138 over 10 seeded inferences, against N1.5's bound of 0.026: `vlm_graphs_autotune` and
`llm_graphs_mpad_autotune`. Bug (an overwritten CUDA-graph output) or a numeric change?

**Method.** `bench/groot_part_diag.py`. One process per rung, with seeded noise matching eager (`fallback_random`).
After 20 warm-up inferences, 3 inferences with `torch.manual_seed(5000 + i)` record the output of every part. Relative
L2 vs `eager_fp32` (`raw/parts_<rung>.npz`, `raw/log_<rung>.txt`):

| rung | vision | LLM hidden | backbone features | actions (3 seeds) |
|---|---|---|---|---|
| `ae_graphs_mpad` (head compiled, backbone eager) | 0 | 0 | 0 | 2.5e-3 |
| `llm_graphs` | 0 | 1.51e-2 | 1.51e-2 | 5.9e-3 |
| `llm_graphs_autotune` | 0 | 1.59e-2 | 1.59e-2 | 5.6e-3 |
| `llm_graphs_mpad_autotune` | 0 | 1.73e-2 | 1.73e-2 | 1.1e-2 |
| `vis_compile` | **4.60e-2** | 7.55e-2 | 7.55e-2 | 1.2e-2 |
| `vlm_graphs_autotune` | **4.15e-2** | 7.60e-2 | 7.60e-2 | 1.1e-2 |

Per-seed relative L2 of the search's own 10 inferences, vs the ladder's `eager_fp32`:

| rung | seeds 5000–5009 | all |
|---|---|---|
| `vlm_graphs_autotune` | 0.030 0.024 **0.247** 0.031 0.028 0.027 0.032 0.025 0.027 0.030 | 0.138 |
| `llm_graphs_mpad_autotune` | 0.040 0.020 **0.247** 0.025 0.021 0.025 0.029 0.025 0.027 0.036 | 0.138 |
| `llm_graphs_autotune` (eligible) | 0.023 0.013 0.008 0.015 0.015 0.017 0.017 0.018 0.019 0.024 | 0.016 |
| `ae_graphs` (compile-only reference) | 0.022 0.020 0.008 0.024 0.020 0.019 0.018 0.019 0.020 0.020 | 0.017 |

**Findings:**
- **Not a CUDA-graph defect.** Every part's difference is identical across calls, which a stale or overwritten graph
  output would not produce. Outputs are finite.
- **Compiling N1.5's vision encoder changes its output by 4.1–4.6 % (relative L2).** That is about three times the
  difference compiling its LLM causes (1.5 %), and it grows to 7.6 % at the backbone output.
- The action head damps most of it (≈ 1e-2 on most seeds). But **one seed (5002) lands in a different action mode**
  (0.247) in both rejected candidates. A perturbation in the features moved that flow trajectory. The 10-seed
  aggregate is dominated by that seed.
- `llm_graphs_mpad_autotune` shows the same jump without compiled vision. The autotuned kernels picked in that
  process gave a larger LLM perturbation than in the diagnostic process (1.7e-2 here). Autotuning picks kernels by
  timing, so its numerics can differ from process to process.

**Decision under T005's fixed rule.** Both candidates stay **not eligible**: relative L2 above the bound, from a
real behavioural difference on one seed, not from rounding. They are not repaired or re-tuned. The next eligible
N1.5 configuration stands.
