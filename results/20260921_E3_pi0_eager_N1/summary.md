# E3 — π0 eager, N = 1 denoising steps

LeRobot π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02`), eager, bf16, batch 1,
MAXN + `jetson_clocks` (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz),
one real LIBERO observation (`HuggingFaceVLA/libero` ep 0
frame 0). 20 warm-up + 100 timed iterations.
Weight integrity verified: 400/400 sampled tensors match the checkpoint.

| region | median ms | p5–p95 | share of e2e |
|---|---|---|---|
| vision | 241.12 | 241.01–241.23 | 44.9 % |
| vlm_prefix | 234.35 | 234.22–234.53 | 43.6 % |
| ae_total (1 steps) | 55.90 | 55.63–56.45 | 10.4 % |
| **end-to-end** | **537.35** | 536.94–538.32 | 100 % |

- **55.90 ms per denoising step** (ae_total / N); analytic floor 3.115 ms → **17.9× floor**.
- P = 816 prefix tokens (533 valid), S = 51 suffix tokens.
- Peak GPU memory 9.04 GB allocated, 9.40 GB reserved.
- Control frequency 1.860 Hz (wall clock, this observation, no simulator).
- `throttled: false` — GPU clock locked min==max for the whole run.

Odd: denoising step 0 costs ~3 ms more than every later step at every N (see `per_step.csv`
at N=10 and `ae_step_individual[i]` rows in `latency_summary.csv`).
