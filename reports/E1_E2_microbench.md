# E1 / E2 — the ceilings of this Thor, the counters that place kernels against them, and the π0 expert's kernels (T003)

Jetson AGX Thor, MAXN (id 0), GPU locked 1575 MHz, EMC 4266 MHz, 20 SMs, `jetson_clocks` + fan; the protocol §1 gate
(incl. GPU idle via `nvidia-smi`, system-wide) passed before every run. 2026-10-22.
Results: `results/20261022_E1_peak/` (ceilings, `hw_constants.json`, `validation.csv`),
`results/20261022_E2_ae_kernels/`, `results/20261022_E1_bw_gemm_stream/`, `results/20261022_E1_load_latency/`.

## The answer first: the ceilings (`results/20261022_E1_peak/hw_constants.json`)

| ceiling | Thor measured | at | NVIDIA spec (Thor) | Orin measured | Thor ÷ Orin |
|---|---|---|---|---|---|
| DRAM bandwidth, best (read-only `x.sum()`) | **258.23 GB/s** | 1 GiB | 273 GB/s (94.6 %) | 187.47 GB/s (axpy) | 1.38 |
| bandwidth, axpy / copy (SM) | 233.69 / 236.10 GB/s | 1 GiB | | 187.47 / 164.47 | 1.25 / 1.44 |
| bf16 dense GEMM | **129.79 TFLOP/s** | 2048³ | not published | 33.85 | 3.83 |
| fp16 | 129.82 TFLOP/s | 2048³ | not published | 32.15 | 4.04 |
| tf32 | 62.37 TFLOP/s | 1024³ | not published | 15.96 | 3.91 |
| fp32 (`highest`, SIMT) | 6.43 TFLOP/s | 16384³ | 8.06 derived (2,560 cores × 2 × 1.575 GHz), 80 % | 3.48 | 1.85 |
| int8 (`torch._int_mm`) | 258.40 TOP/s | 8192³ | not published | 10.50 (not a hardware peak there) | — |
| fp8 e4m3 (record only) | 302.92 TFLOP/s | 4096³ | only "2070 TFLOPS (FP4—Sparse)" | — | — |
| **ridge, bf16** | **502.6 FLOP/B** | | | 180.6 | 2.78 |
| ridge fp32 / tf32 / fp16 / int8 | 24.9 / 241.5 / 502.7 / 1000.7 | | | | |

Spec source: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ (accessed 2026-10-22).
Orin: `reference/orin/reports/E1_E2_microbench.md` ("The answer first" and the E1 tables; Orin, MAXN).

Definitions:
- **bandwidth** = (bytes read + written) ÷ the median per-call time of one repeat (100 calls). The best repeat is
  taken over 3 sizes (1, 2, 4 GiB), 5 repeats and 2 processes.
- **compute** = 2n³ ÷ the median time of the calls before the row's first soctherm over-current event, after a
  cool-down: the *burst* rate.
- **spread**: bandwidth processes agree within ×1.017 (258.23 vs 253.82 GB/s), and repeats at the best size within
  ×1.022. The Orin saw up to ×1.19 between processes; Thor does not.

What the device delivers is also limited in two ways that a single peak hides:
1. **A ≈ 100 W rail limit.** VDD_GPU is current-limited at 5,060 mA (T002). Back-to-back heavy GEMMs settle at
   **93.97 TFLOP/s bf16** (−5.6 %) and keep falling as the die warms (91.9 at 60 s, 80 °C;
   `results/20261022_E0_oc_throttle/`). The sustained rate is in `hw_constants.json` beside the burst ceiling. The
   choice of ceiling is a question in `STATUS.md`.
2. **The L2 is small for this much compute.** bf16 GEMMs reach 130 TFLOP/s only while the operands fit the 32 MiB L2
   (2048³). At 4096³ the GEMM reads **3.57× its A + B bytes from DRAM** (`validation.csv`) and runs at 82 TFLOP/s;
   at 16384³, 58. Same for tf32 (62 → 22–27) and fp8 (303 at 4096³ → 99–101 at 16384³). No over-current event occurred
   in those rows: they are bandwidth-limited.

## Counter validation (`results/20261022_E1_peak/validation.csv`; ncu 2025.3.0, form D, cold caches)

| counter | kernels (known traffic / work) | measured ÷ analytic | used by the roofline |
|---|---|---|---|
| `lts__d_sectors_fill_sysmem.sum` × 32 B | SM copy, axpy, sum, GEMV 8192² | **1.0001–1.0005** | DRAM read bytes |
| `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` × 32 B | same | 1.0002–1.0004 | cross-check |
| `lts__t_sectors_aperture_sysmem_op_write.sum` × 32 B | SM copy, axpy | **1.0006–1.0007** | DRAM write bytes |
| SIMT FP: fadd + fmul + 2·ffma (`smsp__sass_thread_inst_executed_op_*_pred_on.sum`) | fp32 SIMT GEMM 4096³ | **1.0001** | SIMT FLOPs |
| tensor: Σ `sm__ops_path_tensor_src_*.sum` | tf32, int8, fp8 GEMM 4096³ | **1.000** | tensor FLOPs |
| | bf16 / fp16 GEMM 4096³ | 1.0078 = 4128 / 4096 (96-row tiles pad M) | executed, padded work |
| | x [51, 1024] · W [4096, 1024]ᵀ | 1.098 (M = 51 padded to 56) | |
| `mcc__dram_throughput_*` (memory controller) | all | 0 on every kernel | not usable |

Notes:
- There are no `dram__*` metrics on GB10B. The L2 sysmem counters stand in for DRAM bytes and validate to 0.1 %.
- The tensor counter counts the MMA work actually executed, tile padding included, so for skinny shapes measured
  tensor FLOPs exceed useful FLOPs. T006 reports both.
- The 8192² GEMV runs split-K: a second `splitKreduce_kernel` and 16.8× the analytic output writes (partials in a
  workspace). That is real traffic, not a counter error.

## E2 — the π0 action expert, kernel by kernel (`results/20261022_E2_ae_kernels/`)

Real layer weights from the checkpoint (hidden 1024, q 1024→2048, k / v 1024→256, o 2048→1024, gate / up
1024→4096, down 4096→1024; 18 layers, 33.0 MiB per layer). Methods:
- **kernel time** = CUDA-graph replay with "cold" weights (≥ 3× L2 of distinct copies);
- **issue time** = back to back from Python;
- **floors** from weight bytes ÷ 258.23 GB/s (`kernels_floors.csv`).

| at S = 51 | q | k | v | o | gate | up | down | per layer | per step (×18) |
|---|---|---|---|---|---|---|---|---|---|
| kernel, cold (µs) | 19.7 | 4.1 | 4.1 | 18.4 | 48.5 | 48.2 | 38.0 | 181.0 | **3.26 ms** |
| kernel, L2-resident (µs) | 5.5 | 3.0 | 3.0 | 6.0 | 9.1 | 9.1 | 9.9 | 45.6 | 0.82 ms |
| issued from Python (µs) | 20.9 | 12.5 | 12.6 | 19.4 | 49.0 | 48.7 | 38.7 | 201.8 | 3.63 ms |
| ÷ weight floor | 1.21 | 2.03 | 2.03 | 1.13 | 1.49 | 1.48 | 1.17 | | 1.35 (floor 2.41 ms) |
| weight GB/s achieved | 213 | 127 | 127 | 228 | 173 | 174 | 221 | | |

Orin (same E2, `reference/orin/reports/E1_E2_microbench.md` item 3/5): the 126 projections took 4.41 ms per step as
one CUDA graph and 8.66 ms issued from Python, 1.33× their memory floor. Thor: 3.26 ms kernel, 3.63 ms issued.
The CPU no longer doubles the cost on Thor.

**The plan's three E2 questions:**
1. **At which S does a layer stop being launch / latency-bound?**
   - q, o, gate, up and down are weight-streaming-bound at every S from 1 to 256: kernel time is flat over S (q
     19.2–20.5 µs, down 38.0–44.0 µs) and issue ≈ kernel.
   - gate / up get *faster* at S = 128–256 (45.4 → 38.4 µs): cuBLAS switches from a 256×56 to a 144×128 tile and
     streams the weight at 219 GB/s instead of ≈ 175.
   - No projection becomes compute-bound by S = 256: the bf16 compute floor of gate at S = 256 is 16.5 µs, against
     38.4 µs measured.
   - k and v (0.5 MiB) stay **launch-bound at every S**: 4.1–4.3 µs of kernel, 12.3–12.5 µs to issue.
   - Attention against the 816-token prefix is flat to S ≈ 32 (19.8–21.9 µs), then grows: 23.9 at 51, 45.6 at 128,
     65.7 at 256.
2. **What does launching one small kernel cost on this CPU?**
   - 6.4 µs per empty kernel issued back to back; 16.5 µs measured between one event pair.
   - The smallest GEMM issues in 12.2 µs; a q_proj chain fit at S = 51 gives 12.9 µs per call.
   - Unfused single PyTorch ops are issue-bound at S = 51: RMSNorm 65.7 µs to issue vs 16.5 µs of kernel; RoPE
     45.5 vs 17.8 µs.
3. **How much does graph replay save?** For the projections, 0.5–1.2 µs per call (issue 201.8 µs vs kernel 181.0
   µs per layer at S = 51, −10 %). For k / v, 8.4 µs each (−67 %). For RMSNorm / RoPE, 27–49 µs each. The expert
   step's saving is in its non-GEMM operators, not its GEMMs.

The ×3 L2 cold setup matters: with a layer's weights resident (hot), the projections would take 45.6 µs per layer
instead of 181.0. A layer's weights (33 MiB) just exceed the 32 MiB L2, so the real model streams them from DRAM.

## Other measurements
- **Load latency** (`results/20261022_E1_load_latency/`, pointer chase, cycles at 1.575 GHz):
  - L1 ≈ 40 cycles (25 ns);
  - L2 ≈ 255 cycles (162 ns) from 1 to 24 MiB;
  - DRAM ≈ 485–550 cycles median (308–349 ns) at 64–256 MiB, 387 at 1 GiB;
  - repeats beyond L2 vary up to ×2 (cause not measured).
- **Skinny-GEMM weight streaming** (`results/20261022_E1_bw_gemm_stream/`): over 1 GiB of distinct weights (no reuse
  possible), weight bytes ÷ time is 172–219 GB/s. That is 67–85 % of the 258 GB/s ceiling, consistent with E2.
  - The same rows reach 293 GB/s if the L2-resident activations are counted as DRAM bytes, above the 273 GB/s spec.
  - This was my accounting error, now corrected: E2's first `floor_ms` made the same mistake.

## Anomalies and confidence
- High confidence in the bandwidth ceiling (two processes, ×1.017) and in the counter validation (0.1 %).
- The bf16 / fp16 / tf32 / fp8 ceilings are sizes at which the operands fit in L2. They are the right "peak" for a
  roofline, but large GEMMs on this device never see them.
- int8 was not verified as a hardware peak (cuBLASLt `cutlass3x_sm100 … i256x256x32` kernel). It is recorded as
  measured; no model uses it.
- Measured: everything above, except the derived FP32 spec (core count × clock) and the ridge (a ratio of two
  measured values).
- Not measured: power during E1 (T010).

## Reproduce
```bash
source env/thor_env.sh
for p in p1 p2; do for part in bw gemm; do ~/venvs/base/bin/python bench/micro_peak.py --part $part --proc-tag $p --out results/<run>; done; done
for c in copy_sm axpy read_sum gemv skinny_bf16 gemm_{fp32,tf32,bf16,fp16,int8,fp8}; do
  env/rootdo.sh ncu --clock-control none --metrics "$(paste -sd, results/<run>/ncu/validation_metrics.txt)" \
    --nvtx --nvtx-include "val/" --profile-from-start off -o results/<run>/raw/val_$c -f ~/venvs/base/bin/python bench/ncu_validation.py --case $c; done
~/venvs/base/bin/python bench/parse_ncu_validation.py --dir results/<run>
~/venvs/base/bin/python bench/hw_constants_build.py --e1 results/<run> --t002 results/20261022_E0_platform_MAXN/hw_constants.json
HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/micro_ae_kernels.py --hw results/<run>/hw_constants.json --out results/<E2>
~/venvs/base/bin/python bench/e2_floors.py --e2 results/<E2> --hw results/<run>/hw_constants.json
```

## Deviations
- **Compute ceiling.** It is the burst rate (STATUS.md question). The sustained rate is listed beside it.
- **E2 kernel time.** It comes from CUDA-graph replay, not from "CUDA events over 1,000 calls". On this device that
  event chain measures max(kernel, CPU issue), which is reported separately as `chain_ms` and `issue_ms`.
- **E2 cold setup.** "Cold" uses ≥ 3× L2 of weight copies, not the Orin's 18 layers, because Thor's L2 is 32 MiB.
- **E2 floors.** They were recomputed from weight bytes only (`kernels_floors.csv`). The run's own `floor_ms`
  counted activations.
- **Added beyond the task:** the GEMM-stream bandwidth addendum and the E2 ncu check, both prompted by the
  floor-ratio anomaly.

## Self-review (protocol §11)
| check | result |
|---|---|
| `hw_constants.json` complete: bandwidth with spread, TFLOP/s fp32 / tf32 / bf16 / fp16 (+ int8, fp8), ridge, counter validation, each traceable to a CSV row | yes (`row` field per value) |
| `validation.csv` ≥ 4 byte kernels and every dtype's GEMM for FLOPs | yes (4 byte kernels, 6 GEMM dtypes, + skinny) |
| E2: every projection at every S, kernel names, launch cost | yes (`kernels.csv`, `kernels_floors.csv`, `ops.csv`, `launch_cost.csv`) |
| ceilings line of `PROGRESS.md` ticked | yes |
| meta.json complete, raw files hashed, secrets scan clean, no file > 50 MB | yes (4 directories) |
| independent review (P0) | see the T003 status block |
