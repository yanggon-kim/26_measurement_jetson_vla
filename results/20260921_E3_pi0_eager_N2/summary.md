# E3 — π0 eager, N = 2 denoising steps

LeRobot π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02`), eager, bf16, batch 1,
MAXN + `jetson_clocks` (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz),
one real LIBERO observation (`HuggingFaceVLA/libero` ep 0
frame 0). 20 warm-up + 100 timed iterations.
Weight integrity verified: 400/400 sampled tensors match the checkpoint.

| region | median ms | p5–p95 | share of e2e |
|---|---|---|---|
| vision | 240.30 | 239.48–241.14 | 40.6 % |
| vlm_prefix | 234.63 | 233.05–236.06 | 39.6 % |
| ae_total (2 steps) | 110.35 | 109.94–111.60 | 18.6 % |
| **end-to-end** | **592.10** | 589.39–594.20 | 100 % |

- **55.17 ms per denoising step** (ae_total / N); analytic floor 3.115 ms → **17.7× floor**.
- P = 816 prefix tokens (533 valid), S = 51 suffix tokens.
- Peak GPU memory 9.04 GB allocated, 9.40 GB reserved.
- Control frequency 1.689 Hz (wall clock, this observation, no simulator).
- `throttled: false` — GPU clock locked min==max for the whole run.

Odd: denoising step 0 costs ~3 ms more than every later step at every N (see `per_step.csv`
at N=10 and `ae_step_individual[i]` rows in `latency_summary.csv`).
