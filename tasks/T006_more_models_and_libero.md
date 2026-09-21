# T006 — π0.5, SmolVLA, LIBERO closed loop (E3, E4 for more models, E8)  · priority P2 · depends on: T003

1. Repeat T002/T003 (E3 eager, E4 compiled; same regions, sweeps and files) for **π0.5** (openpi, checkpoint
   `pi05_libero`, S = 50) and **SmolVLA** (lerobot at the pinned commit, its published LIBERO / base checkpoint —
   record which; default N = 10, sweep N ∈ {1, 2, 4, 6, 8, 10}). One environment per model. The profiling of these
   models (E5, E5b, E6, E7) is T008; their hybrid-model inputs (E11) are T011, which runs right after step 1 of
   this task; GR00T N1.7 is T009 / T012.
2. E8: LIBERO closed loop per the plan (start with 10 episodes per task). Report success rate, episodes, seeds,
   policy vs simulator time, achieved control Hz, for eager vs compiled and N ∈ {10, 4, 2}.
Acceptance: `reports/E3_E4_models.md` (one table across models: per region ms, ms per denoising step eager and
compiled, floor and gap ratio per protocol §9) and `reports/E8_libero.md`. Self-review (§11).
Raw files (protocol §3.5 / §6): as T002 / T003 per model; for E8 the per-episode, per-step log (success, steps, policy ms,
simulator ms per control step) as CSV. Rollout videos are not committed unless a task asks for them.
