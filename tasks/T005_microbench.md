# T005 — Microbenchmarks (E1, E2)  · priority P1 · depends on: T013 (MAXN; protocol §1 gate)

Implement `bench/micro_peak.py` (E1) and `bench/micro_ae_kernels.py` (E2) exactly as specified in
`docs/01_measurement_plan.md`. Acceptance: `reports/E1_E2_microbench.md` with achieved GB/s and TFLOP/s vs the
204.8 GB/s / 42 TFLOP/s spec, the per-layer floor-multiple vs S table, the measured per-kernel launch cost (µs),
and the effect of CUDA-graph replay.
Also complete `hw_constants.json` (protocol §8) with the measured peaks and the ridge point, and state in the
report how the measured values compare with the expected 204.8 GB/s / ≈ 42 TFLOP/s / ≈ 208 FLOP/B. `bench/micro_ae_kernels.py`
must accept `--layer`, `--S`, `--ncu-window` (warm up, then run the selected kernel between
`cudaProfilerStart/Stop` inside an NVTX range `micro`) so that T007 can use it as an ncu target.
**Substitute validation (protocol §3.3, question 1).** With ncu (filtered, `-c` ≤ 3 launches each, the clock-control
value T013 selected) on four E1 kernels whose traffic is known analytically: (a) `clone()` of a 256 MiB tensor (reads
= writes = 256 MiB), (b) a bf16 GEMV / skinny GEMM that streams an 8–64 MB weight once (reads ≫ writes), (c) the same
kernel launched again immediately (warm L2), (d) a kernel whose working set fits far inside the L2 (expected
≈ 0 misses). Write `ncu/sysmem_miss_validation.csv`: `kernel, analytic_read_bytes, analytic_write_bytes,
miss_read_sectors, miss_write_sectors, miss_all_sectors, measured_read_bytes, measured_write_bytes, ratio_read,
ratio_write, nsys_kernel_ms, GBps_from_misses, GBps_analytic, pct_of_measured_peak`. Report the ratios as found; if
the write path does not appear in any sysmem-miss counter, say so — the substitute then stays read-only.
**Rail double-count check (protocol §4).** Four rails from sysfs at idle, during the ≥ 60 s `clone()` loop and during
the ≥ 60 s 8192² bf16 GEMM loop → `power_rails_check.csv` (`load, rail, mean_mW, above_idle_mW`) and one paragraph.
Raw files (protocol §3.5 / §6): every per-iteration sample and the run stdout committed under the result directories' `raw/`.
