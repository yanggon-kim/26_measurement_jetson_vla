# T006 — π0.5, SmolVLA, LIBERO closed loop (E3, E4 for more models, E8)  · priority P2 · depends on: T003

1. Repeat T002/T003 (E3 eager, E4 compiled; same regions, sweeps and files) for **π0.5** and **SmolVLA**
   (default N = 10, sweep N ∈ {1, 2, 4, 6, 8, 10}). One environment per model. **Route (2026-09-22, after T002's
   accepted route change):** take the LeRobot port first for both — π0.5 as `PI05Policy` / the `pi05` policy of the
   lerobot ref you pin, with its published LIBERO fine-tune if one exists, else its base checkpoint; SmolVLA at the
   same or its own pinned lerobot ref with its published checkpoint — record ref, checkpoint name and sha256, and
   repeat T002's `assert_real_weights()` check (the random-init trap of T002 §8 applies to every LeRobot policy).
   Use openpi's `pi05_libero` only if LeRobot has no loadable π0.5 checkpoint, and then record the conversion.
   Report S, P (processed and valid), expert layers, hidden and KV geometry from the loaded model, as T002 did. The profiling of these
   models (E5, E5b, E6, E7) is T008; their hybrid-model inputs (E11) are T011, which runs right after step 1 of
   this task; GR00T N1.7 is T009 / T012.
2. E8: LIBERO closed loop per the plan (start with 10 episodes per task). Report success rate, episodes, seeds,
   policy vs simulator time, achieved control Hz, for eager vs compiled and N ∈ {10, 4, 2}.
Acceptance: `reports/E3_E4_models.md` (one table across models: per region ms, ms per denoising step eager and
compiled, floor and gap ratio per protocol §9) and `reports/E8_libero.md`. Self-review (§11).
Raw files (protocol §3.5 / §6): as T002 / T003 per model; for E8 the per-episode, per-step log (success, steps, policy ms,
simulator ms per control step) as CSV. Rollout videos are not committed unless a task asks for them.

status: claimed 2026-09-26
status: done 2026-09-26

**Step 1 (E3 / E4 for π0.5 and SmolVLA)** — `reports/E3_E4_models.md` (one table across π0, π0.5,
SmolVLA, GR00T N1.7: per-region ms, ms per denoising step eager and compiled as the §2 slope over
N ∈ {1, 2, 4, 6, 8, 10}, floor and gap ratio per §9). Results: `results/20260925_E3E4_pi05_Nsweep`,
`results/20260925_E3E4_smolvla_Nsweep` (both `meta.json` corrected in d3d4591: the writer recorded π0's
checkpoint for every model; data unaffected), the Part B ladders `results/20260922_E4c_{pi05,smolvla}_opt_ladder`.
Per-step: π0.5 61.60 → 18.40 ms, SmolVLA 61.46 → 4.33 ms (eager_tf32 → best rung).

**Step 2 (E8 LIBERO closed loop)** — `reports/E8_libero.md`; 11 result directories
`results/20260925_E8_pi0_*_settle10*`, each with `summary.json`, `episodes.csv`, `meta.json`, and the
per-step CSV `raw/steps.csv.zst` (all but one, which was re-run for it). Headline: `bf16_both` 249 / 400
vs stock 252 / 400 on the four suites (paired p = 0.87); `libero_spatial` N = 10 / 4 / 2: 62 / 74 / 85.

**Deviations, stated:**
* Owner's **Core** scope (2026-09-25): π0 only; stock and `bf16_both` at N = 10 on 4 suites; `bf16_both`
  only at N ∈ {4, 2}, and only on `libero_spatial`. π0.5 not run in closed loop: LeRobot has only a base
  checkpoint, which scored 0 / 30 (`results/20260923_E8q_pi05_libero_fp32`). SmolVLA not run (outside Core).
* The first settle10 arm has no per-step log (harness fixed mid-run, 0701d95, additive); re-run as
  `…_steplog`. The re-run also measured run-to-run spread: 63 vs 56 / 100 — π0's flow noise is on an
  unseeded torch RNG, so episode outcomes are not seed-determined.
* "fp32" in directory names = the checkpoint as shipped (3.61 B bf16 + 0.42 B fp32 parameters).
* Instrumented per-step (§2 (a)) not available for π0.5 / SmolVLA; the slope (§2 (b)) is used.

bench commit `b2f58f79b27b`.
