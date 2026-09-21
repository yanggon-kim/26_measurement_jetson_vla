# E3 — π0 eager, N = 10 denoising steps

LeRobot π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02`), eager, bf16, batch 1,
MAXN + `jetson_clocks` (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz),
one real LIBERO observation (`HuggingFaceVLA/libero` ep 0
frame 0). 20 warm-up + 100 timed iterations.
Weight integrity verified: 400/400 sampled tensors match the checkpoint.

| region | median ms | p5–p95 | share of e2e |
|---|---|---|---|
| vision | 239.60 | 239.45–239.75 | 23.7 % |
| vlm_prefix | 232.93 | 232.59–233.21 | 23.0 % |
| ae_total (10 steps) | 533.03 | 532.23–537.23 | 52.7 % |
| **end-to-end** | **1011.57** | 1010.56–1016.23 | 100 % |

- **53.30 ms per denoising step** (ae_total / N); analytic floor 3.115 ms → **17.1× floor**.
- P = 816 prefix tokens (533 valid), S = 51 suffix tokens.
- Peak GPU memory 9.04 GB allocated, 9.40 GB reserved.
- Control frequency 0.987 Hz (wall clock, this observation, no simulator).
- `throttled: false` — GPU clock locked min==max for the whole run.

Odd: denoising step 0 costs ~3 ms more than every later step at every N (see `per_step.csv`
at N=10 and `ae_step_individual[i]` rows in `latency_summary.csv`).
