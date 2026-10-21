# T011 — LIBERO closed loop (E8) and the seeded N sweep (E8s) · priority P2 · depends on: T005

## Why
Latency matters only if the optimised policy still works. E8 checks that `best_bf16` does not degrade the closed-loop
success rate against `eager_fp32`, and gives the achieved control frequency on Thor; E8s repeats the Orin's paired,
seeded N sweep, where fewer denoising steps did not lose success.

## Orin counterpart
`reference/orin/tasks/T006_more_models_and_libero.md` step 2, `T016_prefix_bandwidth_and_followups.md` step 2;
`reference/orin/reports/E8_libero.md`, `E8s_libero_seeded.md`; scripts `reference/orin/bench/libero_eval.py`,
`e8s_analyze.py`; `reference/orin/env/setup_libero.sh`.

## Configuration
LIBERO @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` in its own environment (T001), `MUJOCO_GL=egl`. π0 first, then
π0.5 and SmolVLA (GR00T only if a LIBERO harness for it runs: NVIDIA's `GR00T-N1.7-LIBERO` checkpoint, recorded,
optional). **Input:** the checkpoint's empty-camera slot is fed **masked** (`--empty-camera masked`), the training
input; the Orin's E8 / E8s fed it as an attended zero image, so its absolute rates are for the wrong input and only
its paired differences stand. 10 tasks × 10 episodes per suite, `env.seed(seed + episode)`, 10 settle steps.

## Steps
1. **E8**: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; `eager_fp32` vs `best_bf16` per model, the
   same seeds and noise in both arms. Per suite and pooled: successes, Wilson 95 % intervals, paired McNemar (exact),
   policy time vs simulator + rendering time per control step, achieved control Hz.
2. **E8s**: flow noise from a `torch.Generator` seeded per (suite, task, episode), the same seed in every arm;
   `best_bf16`, N ∈ {10, 2, 4, 1} (in that order if time is short) on all four suites; the repeat control (N = 10 on
   `libero_spatial` twice with the same seeds: episode agreement). Seeds recorded per episode in `episodes.csv`.
3. Every action finite-checked (protocol §15); a non-finite episode is reported, never dropped.

## Result directories and reports
`results/<date>_E8_<model>_<suite>_<mode>/`, `results/<date>_E8s_<model>_<suite>_N<n>_seeded/` (`episodes.csv`
per-episode and `steps.csv` per-step logs: success, steps, policy ms, simulator ms); `reports/E8_libero.md`,
`reports/E8s_libero_seeded.md`. Rollout videos are not committed.

## Acceptance
- π0 (at least) on all four suites in both modes with paired statistics; the E8s arms N = 10 and N = 2 on all four
  suites plus the repeat control; the other N and models as time allows, the missing ones named.

## Push points
After each suite and arm, at the close.

## Status (appended by the on-device agent)
