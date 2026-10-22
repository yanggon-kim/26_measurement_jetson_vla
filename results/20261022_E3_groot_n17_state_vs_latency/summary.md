# GR00T N1.7: expert time varies run to run between ≈ 71 and ≈ 89 ms per call (T004 follow-up) — 2026-10-22, MAXN, gate passed

`ae_total` (4 denoising steps) per iteration, counts per 2-ms bin. Every row is a committed file:
- the E3 directories;
- `raw/repeats/` (three plain re-runs of `bench/run_e3.py --steps 4`, copied from `scratch/`);
- the two `state_vs_latency_*.csv` here (`memory_state_probe()` before every iteration).

| run | n | <72 | 72–74 | 74–76 | 76–78 | 78–80 | 80–82 | 82–84 | 84–86 | 86–88 | ≥88 | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3 eager_fp32 (results/20261022_E3_groot_n17_eager_Nsweep) | 100 | 0 | 0 | 0 | 0 | 1 | 18 | 33 | 44 | 3 | 1 | 83.9 |
| shipped (results/20261022_E3_groot_n17_shipped_N4) | 100 | 12 | 70 | 8 | 6 | 4 | 0 | 0 | 0 | 0 | 0 | 72.2 |
| t004_n17_repeat_eager_fp32_083735 | 100 | 1 | 5 | 1 | 1 | 4 | 6 | 1 | 35 | 40 | 6 | 85.8 |
| t004_n17_repeat_eager_fp32_083952 | 100 | 1 | 9 | 1 | 1 | 6 | 6 | 6 | 42 | 24 | 4 | 85.1 |
| t004_n17_repeat_shipped_083844 | 100 | 44 | 12 | 0 | 13 | 7 | 16 | 8 | 0 | 0 | 0 | 72.4 |
| per-iteration probe, eager_fp32 | 150 | 22 | 3 | 0 | 106 | 18 | 0 | 1 | 0 | 0 | 0 | 77.1 |
| per-iteration probe, shipped | 150 | 0 | 79 | 25 | 0 | 22 | 22 | 0 | 1 | 1 | 0 | 74.0 |

- **The timing is not bimodal in a fixed way.** It clusters near ≈ 72, ≈ 77 and ≈ 81–88 ms. Within one run the
  samples usually concentrate in one or two clusters, but the dominant cluster changes from run to run.
- **Precision setting:** shipped runs (torch defaults, cuDNN TF32 on) landed mostly in the low clusters,
  eager_fp32 runs mostly in the high ones, with exceptions.
- **The per-iteration probe changed the picture.** With it, eager_fp32 sat mostly at ≈ 77 ms.
- **It is not the T003 memory-path state:** the probe read "normal" (≈ 49–50 µs) before every iteration in both
  probe runs.
- **The cause is not established.** T006's nsys captures of N1.7 compare kernel timings across iterations.
- **Consequence for the baseline:** N1.7's expert time and share carry a run-to-run uncertainty of about −15 % / +5 %
  around the reported median (83.9 ms, 50.5 %). The v2 sweep (`results/20261022_E3_groot_n17_eager_Nsweep_v2/`) adds
  one more sample of it.
- **Other models:** none shows comparable variation; the regions' p95 / p5 are given in each E3 `summary.md`.
