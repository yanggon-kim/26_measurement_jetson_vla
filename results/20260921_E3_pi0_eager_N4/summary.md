# E3 — π0 eager, N = 4 denoising steps

LeRobot π0 (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02`), eager, bf16, batch 1,
MAXN + `jetson_clocks` (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz),
one real LIBERO observation (`HuggingFaceVLA/libero` ep 0
frame 0). 20 warm-up + 100 timed iterations.
Weight integrity verified: 400/400 sampled tensors match the checkpoint.

| region | median ms | p5–p95 | share of e2e |
|---|---|---|---|
| vision | 239.35 | 239.22–239.74 | 34.5 % |
| vlm_prefix | 233.08 | 232.62–234.56 | 33.6 % |
| ae_total (4 steps) | 215.53 | 214.64–218.83 | 31.1 % |
| **end-to-end** | **694.05** | 692.69–699.49 | 100 % |

- **53.88 ms per denoising step** (ae_total / N); analytic floor 3.115 ms → **17.3× floor**.
- P = 816 prefix tokens (533 valid), S = 51 suffix tokens.
- Peak GPU memory 9.04 GB allocated, 9.40 GB reserved.
- Control frequency 1.438 Hz (wall clock, this observation, no simulator).
- `throttled: false` — GPU clock locked min==max for the whole run.

Odd: denoising step 0 costs ~3 ms more than every later step at every N (see `per_step.csv`
at N=10 and `ae_step_individual[i]` rows in `latency_summary.csv`).
