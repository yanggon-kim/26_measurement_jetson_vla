# Fixing π0's slow MLP gate/up GEMM (T005 diagnostic) — 2026-10-22, MAXN, jetson_clocks, gate passed

Cause: `results/20261022_E4_prefix_pi0_vs_pi05_diag/`. At M = 816 rows, the gate/up projection
`[M×2048]·[2048×16384]` (bf16) gets a 256x96-tile cuBLASLt kernel at 1.7 ms. Nearby M get 0.46–0.60 ms.

**Remedies on the GEMM alone** (`raw/gemm_fix_<remedy>.json`; CUDA events, median of 50; output compared with the
default call):

| remedy | ms | TFLOP/s | output vs default |
|---|---|---|---|
| default (cuBLASLt heuristic) | 1.703 | 32.1 | — |
| PyTorch TunableOp | 1.744 | 31.4 | identical: on CUDA it had only the `Default` candidate (`raw/tunableop_results*.csv`) |
| Inductor `max-autotune`, SM gate forced (ATEN + TRITON) | 1.139 | 48.1 | identical (a Triton template won) |
| **zero-pad M to 976**, slice (copy included) | **0.605** | 90.6 | **bit-identical** |
| **zero-pad M to 1024**, slice (copy included) | **0.589** | 93.0 | **bit-identical** |
| split 768 + 48, concatenate | 1.236 | 44.3 | identical |
| split 720 + 96, concatenate | 1.173 | 46.7 | identical |

**The fix, `linear_mpad` (`bench/rungs.py`, a labelled prep):**
1. One eager inference records every Linear's (M, K, N) in the language model and the action expert.
2. Per shape, M′ = M … M + 256 (step 16) is timed on this device. The fastest M′ is kept if it beats M by more than
   5 %.
3. Those Linears zero-pad their rows to M′ and keep the first M.

GEMM rows are independent, so the real rows are unchanged.

- For π0 it padded two shapes (`raw/prefix_pi0.json`):
  - gate/up (K = 2048, N = 16384): M 816 → 1024, 1.695 → 0.581 ms;
  - k/v projection (N = 256): M 816 → 880, 0.025 → 0.021 ms.
- No expert shape gained more than 5 %.
- π0's prefill replay (`eager_bf16` + `linear_mpad`): **82.74 ms**, against 123.89 ms without (−33 %), and 97.98 ms
  for π0.5.
- Whole model (`results/20261022_E4i_pi0_best_search/`):

| rung (N = 10) | e2e ms | vision | prefix | expert | actions |
|---|---|---|---|---|---|
| `eager_bf16` (ladder) | 293.72 | 28.82 | 125.50 | 136.55 | — |
| `eager_bf16_mpad` | 245.53 | 28.63 | 85.25 | 128.78 | **bit-identical to `eager_bf16`** (10 seeded inferences) |
| `bf16_both_graphs` (ladder) | 211.71 | 21.05 | 108.92 | 79.13 | — |
| `bf16_both_graphs_mpad` | **170.86** | 21.35 | **64.13** | 82.53 | **bit-identical to `bf16_both_graphs`** |
