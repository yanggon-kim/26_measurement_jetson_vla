# T005 — Microbenchmarks (E1, E2)  · priority P1 · depends on: T001

Implement `bench/micro_peak.py` (E1) and `bench/micro_ae_kernels.py` (E2) exactly as specified in
`docs/01_measurement_plan.md`. Acceptance: `reports/E1_E2_microbench.md` with achieved GB/s and TFLOP/s vs the
204.8 GB/s / 42 TFLOP/s spec, the per-layer floor-multiple vs S table, the measured per-kernel launch cost (µs),
and the effect of CUDA-graph replay.
Also complete `hw_constants.json` (protocol §8) with the measured peaks and the ridge point, and state in the
report how the measured values compare with the expected 204.8 GB/s / ≈ 42 TFLOP/s / ≈ 208 FLOP/B. `bench/micro_ae_kernels.py`
must accept `--layer`, `--S`, `--ncu-window` (warm up, then run the selected kernel between
`cudaProfilerStart/Stop` inside an NVTX range `micro`) so that T007 can use it as an ncu target.
Raw files (protocol §3.5 / §6): every per-iteration sample and the run stdout committed under the result directories' `raw/`.
