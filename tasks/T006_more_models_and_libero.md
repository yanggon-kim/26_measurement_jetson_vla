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
