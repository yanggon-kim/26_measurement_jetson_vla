# T003 — Peaks and the roofline ceilings (E1), counter validation, expert kernel sweep (E2) · priority P0 · depends on: T002

## Why
The roofline needs this device's own ceilings: measured memory bandwidth, measured TFLOP/s per dtype, and the ridge
between them. Every later task reads them from the `hw_constants.json` this task writes; none uses a spec sheet or
an Orin value as a ceiling. The DRAM-byte and FLOP counters that place each kernel on the roofline are validated
here, on kernels with known traffic and work. E2 shows where the expert's skinny GEMMs sit against those ceilings.

## Orin counterpart
`reference/orin/tasks/T005_microbench.md`, `reference/orin/reports/E1_E2_microbench.md`; scripts
`reference/orin/bench/micro_peak.py`, `micro_ae_kernels.py`, `hw_constants.py`, `ncu_validation.py`,
`parse_ncu_validation.py`.

## Configuration
Maximum-performance mode, `jetson_clocks` applied, gate of protocol §1 passed. Base environment of T001.

## Steps
1. **E1 bandwidth** (plan E1.1): copy, axpy and read-only reduction on ≥ 1 GiB at ≥ 3 sizes; ≥ 5 repeats in each of
   ≥ 2 processes; per kernel median and spread; the best is `measured_bw_GBps.best`. Say which kernels are SM
   kernels and which run on the copy engine (nsys or the profiler's kernel list).
2. **E1 compute** (plan E1.2): square GEMMs per dtype — fp32 (`highest`), TF32, bf16, fp16, int8 (`torch._int_mm`
   if available), FP8 (`torch._scaled_mm`, record only); n ∈ {1024, 2048, 4096, 8192, 16384 where memory allows};
   ≥ 20 warm-up, ≥ 100 timed; the kernel name picked; the best n per dtype is the peak. Also one 4-dtype run with
   `torch.backends.cuda.matmul` flags printed, so that TF32 is explicit.
3. **Counter validation** (protocol §3.3): with the metric names T001 found, ncu (form D) on: a streaming copy, an
   axpy, a large GEMV (`[1, 8192] × [8192, 8192]` bf16), a large GEMM (4096³ in each dtype). Per kernel: measured
   ÷ analytic DRAM read bytes and write bytes; measured ÷ analytic FLOPs (SIMT and tensor separately, per dtype).
   Write `validation.csv` and state which byte counter and which FLOP counter the roofline will use, and with
   which correction or caveat. If no tensor FLOP counter validates within 5 %, say so: T006 then uses analytic tensor
   FLOPs beside SIMT counters.
4. **Load latency (optional, cheap):** a pointer-chasing kernel over a buffer ≫ L2 and one that fits in L2; ns and
   cycles per dependent load. Labelled measured; it replaces the `[ASSUMPTION]` cycle costs of docs/03 §9 where used.
5. **`hw_constants.json`** (protocol §8): the static fields of T002 plus `measured_bw_GBps`, `measured_tflops`,
   `ridge_flop_per_byte` per dtype, `spec_*` with sources where NVIDIA publishes them, `counter_validation`,
   `source_runs`. Compare with the spec bandwidth and with the Orin's measured peaks (from
   `reference/orin/reports/E1_E2_microbench.md`, labelled Orin) in the report.
6. **E2** (plan E2): every π0 expert projection over S ∈ {1, 2, 4, 8, 16, 32, 51, 64, 128, 256}, attention at
   S = 51 vs 867 keys (SDPA, 1 KV head), RMSNorm, RoPE, GeLU, softmax, add; kernel time (CUDA events over 1,000
   calls), back-to-back time per call from Python, CUDA-graph replay time per call, kernel name, achieved GB/s and
   TFLOP/s, ratio to the bandwidth floor at the measured peak. The launch cost of one small kernel on this CPU (µs).

## Result directories and reports
`results/<date>_E1_peak/` (`latency_summary.csv`, `latency_samples.csv`, `validation.csv`, `ncu/`, `raw/`,
`hw_constants.json`); `results/<date>_E2_ae_kernels/`; `reports/E1_E2_microbench.md` (ceilings table first:
measured vs spec vs Orin; then validation; then E2 with the answers to plan E2's three questions).

## Acceptance
- `hw_constants.json` complete with measured bandwidth (with its spread), TFLOP/s for fp32, TF32, bf16, fp16 (int8
  and FP8 recorded if available), ridge per dtype, and the counter validation; every value traceable to a CSV row.
- `validation.csv` covers ≥ 4 kernels for bytes and the GEMMs of every dtype for FLOPs.
- E2 covers every projection at every S, with kernel names, and the launch-cost answer.
- The ceilings line of `PROGRESS.md` ticked; independent review pass (P0).

## Push points
After `results/<date>_E1_peak/` (the ceilings are what T004–T008 wait for), and at the close.

## Status (appended by the on-device agent)
