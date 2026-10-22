# E4 π0 stock torch.compile modes (T004 step 5) — 2026-10-22, MAXN, jetson_clocks, gate passed, matmul precision `highest` (TF32 off)

One process per (mode, target). e2e = `predict_action_chunk` wall clock, no region instrumentation; 20 warm-up after
compilation, 100 timed per N; ms per step = slope over N ∈ {2, 6, 10}. Eager reference (`results/20261022_E3_pi0_eager_Nsweep/`):
e2e 425.9 ms at N = 10, 12.53 ms per step.

| target | mode | compile (first call) s | graph breaks | e2e N=2 / 6 / 10 ms (median) | ms / step (slope) | peak mem GB (N=10) | CUDA graphs | rel. diff vs eager |
|---|---|---|---|---|---|---|---|---|
| expert | `default` | 22.3 | 1 | 322.55 / 357.19 / 391.9 | 8.669 | 9.19 | no | 4.87e-03 |
| expert | `reduce-overhead` | 21.0 | 1 | 313.84 / 347.03 / 380.14 | 8.288 | 9.19 | yes | 4.87e-03 |
| expert | `max-autotune` | 44.5 | 1 | 319.18 / 351.64 / 383.48 | 8.038 | 9.19 | yes | 4.79e-03 |
| expert | `max-autotune-no-cudagraphs` | 38.8 | 1 | 318.94 / 355.89 / 392.56 | 9.203 | 9.19 | no | 4.76e-03 |
| whole | `default` | 82.8 | 3 | 297.91 / 334.07 / 368.15 | 8.781 | 9.17 | no | 3.21e-03 |
| whole | `reduce-overhead` | 85.1 | 3 | 295.02 / 328.63 / 362.21 | 8.398 | 9.07 | yes | 3.96e-03 |
| whole | `max-autotune` | 164.4 | 3 | 293.19 / 327.54 / 360.71 | 8.439 | 9.07 | yes | 3.33e-03 |
| whole | `max-autotune-no-cudagraphs` | 152.5 | 3 | 305.93 / 341.22 / 376.6 | 8.833 | 9.17 | no | 3.63e-03 |

- **No recompilations** at any N, in any mode. Evidence: the `TORCH_LOGS=recompiles` stream (`[__recompiles]`) printed
  0 lines in every configuration, while the `[__graph_breaks]` stream of the same log printed 33–68. Dynamo's
  `counters["recompiles"]` is never populated in torch 2.10, so the per-N counter columns are n/a (T004 review).
- `whole` graph-breaks 3 times on `copy.deepcopy(past_key_values)` inside `denoise_step` (`modeling_pi0.py:919`;
  Dynamo does not support `deepcopy`). So the denoising loop stays in Python, each step's compiled graph is reused,
  and changing N does not recompile. `expert` breaks once on the same line.
- Inductor logged **"Not enough SMs to use max_autotune_gemm mode"** in every mode: threshold 68 SMs vs 20.
  `max-autotune` therefore autotunes only non-GEMM templates; GEMMs stay with cuBLAS.
- No CUDA-graph skip was logged. Every block was in the normal memory-path state, with no OC events.
- Best: whole `max-autotune`, 360.7 ms at N = 10 (−15.3 % vs eager). Expert per step 8.0–9.2 ms vs 12.53 eager.
  The 276.5–305.2 ms intercept (vision + prefix, uncompiled under `expert`; compiled but graph-broken under `whole`)
  dominates.
