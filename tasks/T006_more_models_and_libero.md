# T006 — π0.5, SmolVLA, LIBERO closed loop (E3–E5 for more models, E8)  · priority P2 · depends on: T003

1. Repeat T002/T003 for π0.5 (openpi) and SmolVLA (lerobot, its default N = 10); GR00T N1.7 if time allows
   (it ships `scripts/activate_orin.sh` and `scripts/deployment/benchmark_inference.py`).
2. E8: LIBERO closed loop per the plan (start with 10 episodes per task). Report success rate, episodes, seeds,
   policy vs simulator time, achieved control Hz, for eager vs compiled and N ∈ {10, 4, 2}.
Acceptance: `reports/E3_E4_models.md` (one table across models) and `reports/E8_libero.md`.
