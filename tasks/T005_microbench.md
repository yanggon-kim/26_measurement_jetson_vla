# T005 — Microbenchmarks (E1, E2)  · priority P1 · depends on: T001

Implement `bench/micro_peak.py` (E1) and `bench/micro_ae_kernels.py` (E2) exactly as specified in
`docs/01_measurement_plan.md`. Acceptance: `reports/E1_E2_microbench.md` with achieved GB/s and TFLOP/s vs the
204.8 GB/s / 42 TFLOP/s spec, the per-layer floor-multiple vs S table, the measured per-kernel launch cost (µs),
and the effect of CUDA-graph replay.
