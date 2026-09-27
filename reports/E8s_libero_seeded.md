# E8s — seeded LIBERO N sweep: does N = 2 beat N = 10 when the noise is paired? (T016 step 2, rows 5–6)

π0 `lerobot/pi0_libero_finetuned_v044`, rung `bf16_both` (as E8), Jetson AGX Orin 64 GB at MAXN + `jetson_clocks`,
2026-09-26/27. All four suites, N ∈ {1, 2, 3, 4, 6, 10}, 10 tasks × 10 episodes per arm, plus the N = 10 repeat on
`libero_spatial`: **25 arms, 2 500 episodes, 0 non-finite action elements**, every arm's per-episode noise seed
checked equal to its N = 10 pair.

## Pre-registered rows, answered

| # | quantity | measured | band |
|---|---|---|---|
| 5 | seeded N = 2 vs N = 10 | N = 2 ≥ N = 10 in **4 of 4** suites (80/71, 88/78, 88/78, 44/32); pooled **300 vs 259 / 400**, discordant 64 / 23, exact McNemar **p = 1.3 × 10⁻⁵** in favour of N = 2; paired difference **+10.2 pp [+5.8, +14.7]** | **replicated** |
| 6 | seeded repeat control (`libero_spatial` N = 10 twice) | 71 and 70 / 100; **95 / 100 episodes agree** (only first 3, only second 2, p = 1) | **≥ 95 %: seeding pairs the arms** — exactly on the line, no margin |

For comparison, E8's *unseeded* repeat agreed on 65 % of episodes (E8 §4): seeding the flow noise removes most of
the run-to-run spread; the residual 5 % is GPU / compiled-kernel non-determinism.

## 1. Pooled over the four suites, each N vs N = 10 (paired on task × episode, same noise seeds)

| N | suites | successes [Wilson 95 %] | N = 10 | only arm / only N10 | exact McNemar p | paired diff (pp) [95 %] |
|---:|---:|---|---:|---:|---:|---|
| 1 | 4 | **295/400** [69.2, 77.8] | 259 | 76 / 40 | 0.00107 | +9.0 [+3.8, +14.2] |
| 2 | 4 | **300/400** [70.5, 79.0] | 259 | 64 / 23 | 1.27e-05 | +10.2 [+5.8, +14.7] |
| 3 | 4 | **286/400** [66.9, 75.7] | 259 | 52 / 25 | 0.0028 | +6.8 [+2.5, +11.0] |
| 4 | 4 | **284/400** [66.4, 75.2] | 259 | 43 / 18 | 0.00187 | +6.2 [+2.5, +10.0] |
| 6 | 4 | **267/400** [62.0, 71.2] | 259 | 23 / 15 | 0.256 | +2.0 [-1.0, +5.0] |
| 10 | 4 | **259/400** [60.0, 69.2] | — | — | — | — |

Pooled success falls from 74–75 % at N = 1–2 to 71–72 % at N = 3–4, 67 % at N = 6 and 65 % at N = 10. N = 1–4 are
each better than N = 10 (p ≤ 0.003); N = 6 is not distinguishable from N = 10.

## 2. Per suite

| suite | N | successes / 100 [Wilson 95 %] | vs N = 10: only arm / only N10 | exact McNemar p | paired diff (pp) [95 %] | agreement with N = 10 | forward ms (median) |
|---|---:|---|---:|---:|---|---:|---:|
| libero_spatial | 1 | **85** [76.7, 90.7] | 20 / 6 | 0.00936 | +14.0 [+4.4, +23.6] | 0.74 | 274.64 |
| libero_spatial | 2 | **80** [71.1, 86.7] | 16 / 7 | 0.0931 | +9.0 [-0.2, +18.2] | 0.77 | 291.72 |
| libero_spatial | 3 | **73** [63.6, 80.7] | 12 / 10 | 0.832 | +2.0 [-7.2, +11.2] | 0.78 | 309.37 |
| libero_spatial | 4 | **76** [66.8, 83.3] | 9 / 4 | 0.267 | +5.0 [-2.0, +12.0] | 0.87 | 326.07 |
| libero_spatial | 6 | **70** [60.4, 78.1] | 4 / 5 | 1 | -1.0 [-6.9, +4.9] | 0.91 | 360.07 |
| libero_spatial | 10 | **71** [61.5, 79.0] | — | — | — | — | 427.53 |
| libero_spatial | 10 rep | **70** [60.4, 78.1] | 2 / 3 | 1 | -1.0 [-5.4, +3.4] | 0.95 | 428.19 |
| libero_object | 1 | **83** [74.5, 89.1] | 18 / 13 | 0.473 | +5.0 [-5.9, +15.9] | 0.69 | 274.67 |
| libero_object | 2 | **88** [80.2, 93.0] | 14 / 4 | 0.0309 | +10.0 [+1.9, +18.1] | 0.82 | 291.91 |
| libero_object | 3 | **82** [73.3, 88.3] | 9 / 5 | 0.424 | +4.0 [-3.3, +11.3] | 0.86 | 309.99 |
| libero_object | 4 | **83** [74.5, 89.1] | 8 / 3 | 0.227 | +5.0 [-1.4, +11.4] | 0.89 | 326.48 |
| libero_object | 6 | **82** [73.3, 88.3] | 6 / 2 | 0.289 | +4.0 [-1.5, +9.5] | 0.92 | 359.87 |
| libero_object | 10 | **78** [68.9, 85.0] | — | — | — | — | 426.78 |
| libero_goal | 1 | **80** [71.1, 86.7] | 14 / 12 | 0.845 | +2.0 [-8.0, +12.0] | 0.74 | 274.65 |
| libero_goal | 2 | **88** [80.2, 93.0] | 14 / 4 | 0.0309 | +10.0 [+1.9, +18.1] | 0.82 | 292.02 |
| libero_goal | 3 | **91** [83.8, 95.2] | 15 / 2 | 0.00235 | +13.0 [+5.3, +20.7] | 0.83 | 308.59 |
| libero_goal | 4 | **85** [76.7, 90.7] | 10 / 3 | 0.0923 | +7.0 [+0.1, +13.9] | 0.87 | 326.03 |
| libero_goal | 6 | **81** [72.2, 87.5] | 4 / 1 | 0.375 | +3.0 [-1.3, +7.3] | 0.95 | 361.88 |
| libero_goal | 10 | **78** [68.9, 85.0] | — | — | — | — | 425.35 |
| libero_10 | 1 | **47** [37.5, 56.7] | 24 / 9 | 0.0135 | +15.0 [+4.1, +25.9] | 0.67 | 274.8 |
| libero_10 | 2 | **44** [34.7, 53.8] | 20 / 8 | 0.0357 | +12.0 [+1.9, +22.1] | 0.72 | 291.91 |
| libero_10 | 3 | **40** [30.9, 49.8] | 16 / 8 | 0.152 | +8.0 [-1.5, +17.5] | 0.76 | 309.2 |
| libero_10 | 4 | **40** [30.9, 49.8] | 16 / 8 | 0.152 | +8.0 [-1.5, +17.5] | 0.76 | 326.42 |
| libero_10 | 6 | **34** [25.5, 43.7] | 9 / 7 | 0.804 | +2.0 [-5.8, +9.8] | 0.84 | 359.31 |
| libero_10 | 10 | **32** [23.7, 41.7] | — | — | — | — | 428.64 |

Per-suite p values are unadjusted (24 comparisons); the pre-registered test is the pooled N = 2 one. Forward =
a `select_action` call that ran the model (> 10 ms); median over the arm's ~300–900 forwards.

## 3. Reading (derived)

* **The E8 finding replicates with paired noise, on every suite, but smaller.** E8 (unseeded, spatial only) had
  N = 2 vs N = 10 at 85 vs 62 (+23 pp); seeded, spatial is 80 vs 71 (+9 pp, p = 0.09 on that suite alone) and the
  pooled effect is +10.2 pp. E8's unseeded N = 10 spatial arm (62) and this seeded one (71) differ by 9 episodes —
  of the order of E8's unseeded run-to-run spread (7), so part of E8's +23 was noise.
* **Shape:** the gain is largest at N = 1–2, shrinks through N = 3–4 (+6–7 pp) and is gone by N = 6. No suite has
  N = 10 as its best arm; the best arm is N = 1 (spatial, libero_10), N = 2 (object) or N = 3 (goal), all within
  each other's intervals.
* **Agreement with N = 10 rises with N** (pooled 0.71 at N = 1 → 0.90 at N = 6 vs 0.95 for N = 10 against itself):
  with the same initial noise, fewer integration steps give measurably different trajectories, not the same ones
  with noise on top.
* **Latency:** forward 274.7 / 291.9 / 309.3 / 326.2 / 360.3 / 427.1 ms at N = 1 / 2 / 3 / 4 / 6 / 10 (suite medians
  averaged) — linear fit 16.9 ms per denoising step + 258.2 ms (prefix + fixed part); N = 10 → 2 saves 135 ms per
  forward. At this rung the prefix dominates.
* **Not explained:** why fewer steps succeed more often is measured, not tested — candidates (smoother, lower-variance
  chunks from a short integrator; the fine-tune's training schedule) are untested. One precision (`bf16_both`) only.

## 4. Harness and conditions

* `bench/libero_eval.py` (E8's harness; `--seed-noise` added in `cf62176`): per (suite, task, episode) a
  `torch.Generator` seeded with `(suite_idx·10⁶ + task·10³ + episode)·7919 + seed` draws noise of shape
  `(1, chunk_size, max_action_dim)`, passed to `policy.predict_action_chunk(noise=…)` — the same noise at every N.
  `env.seed(seed + episode)`, seed 0, 10 no-op settle steps, suite `max_steps` from LeRobot; seed per episode in
  `episodes.csv`. Actions finite-checked on the device (count read once per episode, outside any timing).
* Run order (one GPU job at a time): N = 10 and 2 on all suites, the repeat, then N = 4, 1, 3, 6 — every planned
  arm ran; none missing. MAXN, GPU 1300.5 MHz, per-arm device state in each `meta.json`.
* Result dirs: `results/20260926_E8s_pi0_<suite>_N<n>_seeded` (24), `…_libero_spatial_N10_seeded_rep`; tables
  and statistics `results/20260926_E8s_pi0_seeded_summary/` (`bench/e8s_analyze.py`: Wilson score interval, exact
  two-sided McNemar on discordant pairs, Wald interval for the paired difference).
* The seeded paired π0 `nopad` arm (T016 step 3) uses the same harness and is reported in `E14_prefix_savings.md`.

## Reproduce

```
MUJOCO_GL=egl ~/venvs/libero/bin/python bench/libero_eval.py --model pi0 --rung bf16_both --suite <suite> \
    --tasks 10 --episodes 10 --steps <N> --seed-noise --outdir results/<date>_E8s_pi0_<suite>_N<N>_seeded
python3 bench/e8s_analyze.py results/20260926_E8s_pi0_seeded_summary
```
