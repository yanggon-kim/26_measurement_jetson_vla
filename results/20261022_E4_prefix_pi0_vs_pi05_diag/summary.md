# Why π0's VLM prefix is slower than π0.5's (T005 diagnostic) — 2026-10-22, MAXN, jetson_clocks, gate passed

**Question.** π0's prefix (816 tokens) took 125.5 ms and π0.5's (968 tokens) 96.3 ms in the `eager_bf16` rung. They
share the same Gemma-2B language model and the same prefix code (`modeling_pi0.py` / `modeling_pi05.py`
`embed_prefix` and `sample_actions` are identical). Both run bf16 weights and bf16 activations: `PiGemmaModel.forward`
casts its input to bf16 (`pi_gemma.py`), and both use eager attention.

**Answer: one GEMM shape.** The MLP gate/up projection `[M×2048]·[2048×16384]` (36 calls per prefix) gets a much
slower cuBLASLt kernel at M = 816 than at M = 968.

| replay of the prefill call alone (`raw/prefix_<m>.json`, CUDA events, n = 50) | π0 (M = 816) | π0.5 (M = 968) |
|---|---|---|
| median ms | 123.89 | 97.98 |

Isolated GEMM, CUDA events, median of 50 (`raw/gemm_m_sweep_*.csv`):

| shape | M = 768 | M = 816 (π0) | M = 968 (π0.5) |
|---|---|---|---|
| gate/up `[M×2048]·[2048×16384]` | 0.455 ms, 113 TFLOP/s (128x192) | **1.738 ms, 31.5 TFLOP/s (256x96)** | 0.598 ms, 109 TFLOP/s (112x128) |
| down `[M×16384]·[16384×2048]` | 1.391 ms, 37 TFLOP/s | 1.435 ms, 38 TFLOP/s | 1.357 ms, 48 TFLOP/s |
| q/o `[M×2048]·[2048×2048]` | — | 0.069 ms, 100 TFLOP/s | 0.090 ms, 90 TFLOP/s |

- **The gate/up time depends on M in steps, not smoothly.** Sweep M = 512–1024, step 16:
  - M = 688–704: 4.0 ms (12 TFLOP/s);
  - M = 720–768: 0.46 ms (105–108 TFLOP/s);
  - M = 816–864: 1.74 ms (31–33 TFLOP/s);
  - M = 976–1024: 0.60 ms (110–114 TFLOP/s).
  Each step coincides with a change of the kernel tile cuBLASLt's heuristic picks.
- **What it costs π0:** 36 × 1.74 = 62.6 ms per prefix for gate/up, against 21.5 ms for π0.5's 36 calls at a larger
  M. That accounts for the 26 ms gap and more; π0.5 does more attention work.
- **Not changed by the BLAS backend or the workspace.** `torch.backends.cuda.preferred_blas_library` = cublas or
  cublaslt picks the same kernel. `CUBLASLT_WORKSPACE_SIZE` = 1 MiB or 32 MiB also changes nothing.
- **The down projection is slow at every M** (37–48 TFLOP/s against a measured 129.8 TFLOP/s bf16 peak): K = 16384
  with N = 2048 gives few output tiles for 20 SMs. T006–T008 analyse it.
- Profiler per-op times (`raw/prefix_<m>_ops.csv`, `_kernels.csv`) carry profiler overhead. Use their ratios and
  kernel names; the absolute times above come from CUDA events.

**Consequence for T005.** π0's `best_bf16` search includes Inductor's forced GEMM autotuning. It times candidate
kernels (cuBLAS and Triton) and does not rely on the heuristic. Changing M itself, by trimming the 283 padding tokens,
is a model-input change and belongs to T014 (E14b).
