# E8 — LIBERO closed loop: does π0 still work after the speed-ups? (T006 step 2)

**Answer.** Yes. The full compiled `bf16_both` rung — the configuration behind the study's π0 speed
numbers — succeeds as often as the checkpoint as shipped on all four LIBERO suites: **249 / 400 vs
252 / 400** pooled, every per-suite difference ≤ 2 episodes, paired McNemar p ≥ 0.86 on each suite and
0.87 pooled. Its policy forward is **2.45× faster** in the loop (430.7 vs 1055–1074 ms median) and the
achieved control rate rises from 16–17 Hz to 17–22 Hz. Fewer denoising steps did **not** cost success on
`libero_spatial`: N = 10 / 4 / 2 gave **62 / 74 / 85** of 100.

Scope: the owner's **Core** choice — π0 only, stock and `bf16_both` at N = 10 on all four suites, and
`bf16_both` at N ∈ {4, 2} on `libero_spatial`; 10 tasks × 10 episodes per arm. Resolution: at 100
episodes an arm's 95 % interval is ±9–10 points, and the same arm run twice moved by 7 (§4), so this
resolves a **broken** policy or a large effect, not a few-percent difference.

> **Correction / finding (2026-09-27, T016 step 3).** `bench/libero_eval.py` supplied the checkpoint's
> `observation.images.empty_camera_0` as an **all-zero image**, so LeRobot's `_preprocess_images` treated it as a
> present camera (mask 1, its 256 SigLIP tokens attended). Left out — as in training and in `pi0_common.build_batch`
> — LeRobot synthesises it as a −1 image with **mask 0**. Every arm in this report ran with the empty camera attended.
> Paired comparisons between arms are unaffected (both arms share the input); absolute success rates are for that
> input. `--empty-camera masked` now reproduces the training-time input; see `reports/E14_prefix_savings.md` §1.

## 1. Measured — success

| suite | max steps | stock (as shipped) N=10 | `bf16_both` N=10 | paired: both / only stock / only bf16 / neither | McNemar p |
|---|---:|---:|---:|---|---:|
| libero_spatial | 280 | **63** [53.2, 71.8] | **62** [52.2, 70.9] | 42 / 21 / 20 / 17 | 1.00 |
| libero_object | 280 | **69** [59.4, 77.2] | **70** [60.4, 78.1] | 45 / 24 / 25 / 6 | 1.00 |
| libero_goal | 300 | **80** [71.1, 86.7] | **79** [70.0, 85.8] | 65 / 15 / 14 / 6 | 1.00 |
| libero_10 | 520 | **40** [30.9, 49.8] | **38** [29.1, 47.8] | 22 / 18 / 16 / 44 | 0.86 |
| **pooled** | | **252 / 400** [58.2, 67.6] % | **249 / 400** [57.4, 66.9] % | discordant 78 / 75 | 0.87 |

| `libero_spatial`, `bf16_both` | N = 10 | N = 4 | N = 2 |
|---|---:|---:|---:|
| success / 100 | **62** [52.2, 70.9] | **74** [64.6, 81.6] | **85** [76.7, 90.7] |
| paired vs N = 10 (only N10 / only other) | — | 16 / 28, p = 0.096 | 7 / 30, **p = 0.0002** |
| policy forward, median ms | 430.71 | 329.23 | 294.70 |

Brackets: Wilson 95 % intervals (derived). Paired columns: the same (task, episode) — same LIBERO
init state and `env.seed(seed + episode)` — in both arms; McNemar exact two-sided on the discordant
pairs (derived). Every count is in the arm's `episodes.csv`.

**Stock = the checkpoint as shipped** (`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc`), no rung, no
cast, eager: 3.61 B bf16 + 0.42 B fp32 parameters (`meta.json.param_dtype_numel`). The plan and the
directory names call this arm "fp32"; it is the unmodified baseline, not an all-fp32 model.
**`bf16_both`** = the full rung exactly as the latency study ran it (`opt_ladder.apply_rung`: vision tower,
projector and 74 layernorms to bf16; `model.denoise_step@reduce-overhead`,
`pg.embed_image@reduce-overhead+clone`, `pg.forward@max-autotune-no-cudagraphs`; matmul `high`):
4.02 B bf16 + 3.2 M fp32 parameters.

## 2. Measured — time in the loop

| arm | policy forward median (p95) ms | forwards | sim step median ms | policy share of loop time | typical-step Hz | wall control Hz |
|---|---:|---:|---:|---:|---:|---:|
| spatial stock (run 1 / re-run) | 1058.6 / 1063.5 (p95 re-run 1067) | 385 / 407 | 39.1 | 0.41 ² | 23.3 | 14.1 ² |
| spatial bf16 N=10 | 430.7 (432) | 393 | 38.9 | 0.27 | 23.3 | 17.1 |
| spatial bf16 N=4 | 329.2 (330) | 353 | 39.7 | 0.24 | 22.9 | 17.5 |
| spatial bf16 N=2 | 294.7 (295) | 317 | 40.0 | 0.24 | 22.8 | 17.6 |
| object stock | 1073.6 (1079) | 406 | 26.8 | 0.50 | 32.7 | 16.8 |
| object bf16 | 430.9 (432) | 399 | 26.9 | 0.35 | 32.4 | 21.3 |
| goal stock | 1052.0 (1060) | 319 | 29.2 | 0.47 | 30.3 | 16.3 |
| goal bf16 | 430.0 (432) | 334 | 29.5 | 0.33 | 29.9 | 20.2 |
| libero_10 stock | 1055.4 (1063) | 889 | 26.8 | 0.49 | 32.7 | 17.4 |
| libero_10 bf16 | 430.7 (433) | 904 | 27.1 | 0.33 | 32.2 | 22.2 |

² the first spatial stock run has no per-step log (§5); policy share and wall Hz are the re-run's.

* **Forward** = a `select_action` call that ran the model (> 10 ms); the other 49 of every 50 control steps
  pop the 50-action chunk (`n_action_steps` 50) at ~3.8 ms. The first two `bf16_both` forwards of each arm
  are compile warm-up (24.1 s and 1.0 s on spatial) and are in the p95 only via `steps.csv`.
* **Typical-step Hz** = 1000 / (median sim + median policy call): blind to the forward stalls.
  **Wall control Hz** = control steps / wall time of the policy-acting loop (derived from `steps.csv`): the
  rate the robot would actually see, stalls included. Compiled raises it 16–17 → 17–22 Hz; the simulator
  (27–40 ms per step, CPU) is then the larger share.
* N = 4 → 2 saves only 35 ms per forward and N = 10 → 2 136 ms: the prefix (vision + VLM, ~250 ms at this
  rung, `E12b`) dominates a compiled forward, not the denoising loop.

## 3. N sweep — fewer steps did not hurt (derived reading)

62 → 74 → 85 of 100 is monotone and N = 2 vs N = 10 is beyond the run-to-run spread of §4 (23 vs 7
points; paired p = 0.0002). This is **measured, not explained**: plausible mechanisms (a two-step flow
integrator giving smoother, lower-variance chunks; this checkpoint's fine-tuning favouring few steps)
are not tested here, and only one suite and one precision were swept (owner's Core scope). It does mean
N is not a free latency knob to be traded against accuracy on the evidence of this suite — at N = 2
π0 was both faster and more successful.

## 4. Repeatability — the same arm twice

`libero_spatial` stock, same command, same seeds, same harness except the added per-step log:
**63 / 100** (`…_fp32_N10_settle10`) and **56 / 100** (`…_fp32_N10_settle10_steplog`). Episode-level
agreement 65 % (both succeed 42, only first 21, only second 14; McNemar p = 0.31).

Why outcomes are not seed-determined: `env.seed(seed + episode)` fixes the simulator, but π0's flow
starts from `torch.normal(...)` (`lerobot/policies/pi0/modeling_pi0.py:629 sample_noise`) on torch's
global RNG, which the harness does not seed — every process draws different noise, and bf16 / compiled
kernels are not bit-deterministic either. So: **differences of a few episodes between arms are noise**;
the stock-vs-bf16 gaps (1–3 episodes) are well inside it. Seeding torch per episode would make arms
paired in the noise too; it was not done, so as not to change the harness mid-campaign.

## 5. Harness, conditions, anomalies

* `bench/libero_eval.py`, LIBERO + LeRobot `pi0_libero_finetuned_v044`, `MUJOCO_GL=egl`, 256×256
  cameras; per-suite `max_steps` from `lerobot.envs.libero.TASK_SUITE_MAX_STEPS`; **10 no-op settle steps**
  after `set_init_state`, as LeRobot's own env does (`num_steps_wait=10`; not counted in max_steps).
  MAXN, GPU 1300.5 MHz at start and end of every arm (`meta.json.gate`, `gate_end`); one GPU job at a time
  (10-minute watchdog audit throughout).
* **Superseded, kept**: `results/20260925_E8_pi0_libero_spatial_{fp32,bf16both}_N10` (no settle step:
  spatial stock 18 / 100, bf16 24 / 100) — marked `SUPERSEDED.md`; a stopped no-settle object arm (0 / 88)
  is in `scratch/failed_runs/20260925_E8_pi0_libero_object_fp32_N10_NO_SETTLE_partial`; the earlier quick check
  `results/20260923_E8q_*` (3 episodes × 10 tasks, no settle; π0 stock 6 / 30, bf16 4 / 30).
* **Harness change during E8 (labelled)**: after the first settle10 arm (spatial stock) the harness was
  found to write no per-step log, which T006's raw-file line requires. Fixed in `0701d95` — additive
  only: `raw/steps.csv.zst` (task, episode, step, policy ms, forward flag, sim ms, done), `wall_control_hz`,
  parameter dtypes, `gate_end`. That arm is kept (with `NOTE.md`) and was re-run as `…_steplog` (§4); the
  spatial `bf16_both` arm running at that moment was stopped ~15 min in and moved to
  `scratch/failed_runs/…_STOPPED_no_steplog`.
* **MuJoCo instability warnings**: in each `libero_10` arm, twice, "Nan, Inf or huge value in QACC at
  DOF 9/10 … Time = 0.532/0.536" — the same simulated times in both arms, i.e. seed-determined scene
  events, not policy output (`raw/mujoco_warnings.txt` in both `libero_10` directories; added after the
  run, so not listed in those `meta.json` `raw_files`).
* **π0.5 is not validatable here**: LeRobot publishes only `lerobot/pi05_libero_base` (no LIBERO
  fine-tune); it scored **0 / 30** in the E8q check (`results/20260923_E8q_pi05_libero_fp32`), so it
  cannot serve as a working baseline and was not run (owner's Core scope).

## 6. Not measured

* Stock at N ∈ {4, 2}; `bf16_both` N ∈ {4, 2} on object / goal / libero_10 (outside Core).
* π0.5, SmolVLA, GR00T closed loop (no LIBERO checkpoint that works in this harness / outside Core).
* Why fewer steps succeed more (§3); torch-seeded paired runs (§4).
* Rollout videos (not committed unless a task asks).

## 7. Result directories

`results/20260925_E8_pi0_{libero_spatial,libero_object,libero_goal,libero_10}_{fp32,bf16both}_N10_settle10/`,
`results/20260925_E8_pi0_libero_spatial_bf16both_N{4,2}_settle10/`,
`results/20260925_E8_pi0_libero_spatial_fp32_N10_settle10_steplog/` — each with `summary.json`,
`episodes.csv`, `meta.json` and (all but the first spatial stock arm) `raw/steps.csv.zst`.

Reproduce one arm:

    MUJOCO_GL=egl ~/venvs/libero/bin/python bench/libero_eval.py --model pi0 --suite libero_goal \
        --tasks 10 --episodes 10 --steps 10 [--rung bf16_both] --outdir results/<date>_E8_...

## 8. Self-review (§11)

T006 step 2 asks for success rate, episodes, seeds, policy vs simulator time and achieved control Hz, for
eager vs compiled and N ∈ {10, 4, 2}: all present for π0 within the owner's Core scope (compiled only at
N ∈ {4, 2}; stated). Raw per-episode, per-step CSV: present for 10 of 11 arms, the eleventh re-run for it.
Every derived number (intervals, McNemar, wall Hz, policy share) is recomputed from committed
`episodes.csv` / `steps.csv.zst`. Weakest points: 100 episodes per arm; outcomes not seed-paired in the
policy noise (§4).
