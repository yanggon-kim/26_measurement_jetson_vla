# GR00T N1.7: expert time is bimodal, and not because of the memory-path state (T004 follow-up) — 2026-10-22, MAXN, gate passed

`ae_total` (4 steps) takes either ≈ 72 or ≈ 85 ms per call. Both modes appear under both precision settings
(`eager_fp32`: cuDNN TF32 off; `shipped`: torch defaults, cuDNN TF32 on), and the share of each changes from run to run:

| run | iterations < 78 ms |
|---|---|
| `results/20261022_E3_groot_n17_eager_Nsweep/` (N = 4) | 0 / 100 |
| eager_fp32 repeats (`scratch/`, not kept) | 8 / 100, 12 / 100 |
| `results/20261022_E3_groot_n17_shipped_N4/` | 96 / 100 |
| shipped repeat (`scratch/`) | 69 / 100 |
| this directory, eager_fp32, `memory_state_probe()` before every iteration | 131 / 150 |
| this directory, shipped, same | 104 / 150 |

Here the per-iteration probe read the **normal** memory-path state (≈ 49–50 µs) for every iteration in both modes, so the
hidden state of `results/20261022_E1_state_probe/` does not explain it. Inserting the 3 ms probe changed the mode shares.
The cause is not established. T006's nsys traces will compare the kernels of fast and slow iterations. No other model is
bimodal (p95 / p5 ≤ 1.07 for every region; π0's eager expert has a slow tail, p95 143.7 vs median 126.6 ms).
