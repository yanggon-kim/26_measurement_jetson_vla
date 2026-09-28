## The S sweep: at the model's own S, the time is set by weight bytes, not tokens

> **CORRECTION (2026-09-25, late).** The "median µs", "× floor" and "GB/s" columns of the table
> below came from CUDA events with an event pair **and a synchronize around every single call**,
> which on Orin adds ~45–75 µs per call (found when the π0.5 / SmolVLA sweeps put every small GEMM
> at ~90 µs whatever its shape). They measured the timing harness, not the GEMM. The kernel time is
> in the ncu captures of the same sweep: `results/20260925_E5b_groot_n17_ssweep/s_sweep_ncu.csv`
> (`bench/sweep_ncu_table.py`). Corrected table:
>
> | target | projection (in → out, dtype) | S | kernel µs (ncu) | × floor | GB/s | % measured BW | kernel | events said µs (superseded) |
> |---|---|---:|---:|---:|---:|---:|---|---:|
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 1 | **118.6** | 1.18 | 159.4 | 85.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 166.8 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 16 | **119.3** | 1.17 | 160.4 | 85.6 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 166.1 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 41 | **124.8** | 1.20 | 156.3 | 83.4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 170.2 |
> | `largest` | `ff.net.0.proj` (1536 → 6144, bfloat16) | 256 | **183.1** | 1.28 | 124.6 | 66.5 | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2` | 228.4 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 1 | **37.1** | 1.47 | 127.6 | 68.1 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 93.9 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 16 | **37.1** | 1.44 | 129.9 | 69.3 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_relu_f2f_` | 92.8 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 41 | **38.1** | 1.44 | 130.6 | 69.7 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 93.1 |
> | `q` | `attn1.to_q` (1536 → 1536, bfloat16) | 256 | **82.0** | 2.30 | 76.8 | 41.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f` | 128.9 |
>
> **Readings that change:** the largest projection is at **1.17–1.20× its memory floor** up to
> S = 41 (was quoted 1.6×) — 83–86 % of the measured bus bandwidth, i.e. essentially at the
> weight-streaming limit; the q projection is at **1.44–1.47×** (was 3.5–3.7×), so "too small to
> reach the bus's stride" overstated it. **Readings that stand:** the largest projection costs the
> same from S = 1 to S = 41 (118.6 / 119.3 / 124.8 µs); the kernel choices and stall columns come
> from ncu and are unaffected. The superseded table is kept below as it was written.

DiT block 0's projections were read from the loaded model; the largest is `ff.net.0.proj`
(1536 → 6144) and the q projection `attn1.to_q` (1536 → 1536), both bf16. Each was run alone
as `F.linear` at S ∈ {1, 16, 41, 256}. The table gives the median of 200 reps under CUDA
events, then one ncu capture per point.

| projection | S | median µs | × floor | GB/s | kernel cuBLAS chose | long-scoreboard % of waiting | SM % |
|---|---:|---:|---:|---:|---|---:|---:|
| `ff.net.0.proj` | 1 | 166.80 | 1.65 | 113.3 | `128x64 … stages_32x6` | 20.5 | 25.2 |
| | 16 | 166.08 | 1.63 | 115.2 | `128x64 … stages_32x6` | 20.2 | 25.2 |
| | **41** | **170.24** | **1.64** | 114.6 | **`128x64 … stages_64x3`** | **72.7** | 24.1 |
| | 256 | 228.42 | 1.60 | 99.9 | `128x128 … stages_32x5` | 10.1 | 66.5 |
| `attn1.to_q` | 1 | 93.86 | 3.72 | 50.4 | `64x64_sliced1x2 …` | 53.7 | 20.7 |
| | 16 | 92.83 | 3.61 | 51.9 | `64x64 … stages_64x5` | 51.2 | 21.1 |
| | **41** | **93.07** | **3.51** | 53.4 | `64x64_sliced1x2 …` | 54.0 | 20.1 |
| | 256 | 128.94 | 3.61 | 48.8 | `128x64 … stages_64x3` | 55.6 | 36.8 |

* **The largest projection costs the same from S = 1 to S = 41** (166–170 µs), at 1.6× its
  memory floor. Its time is the weight read; the 41 action tokens are nearly free. That is
  the precise sense in which this action head is a weight-streaming workload.
* **The isolated sweep reproduces the in-model kernel choice.** At S = 41 cuBLAS picks the
  same `128x64 … stages_64x3` kernel that E5b captured inside the model, and it is
  memory-latency bound there too (72.7 %). At S = 256 cuBLAS switches to the compute-bound
  `128x128` tile, the one the eager in-model capture found compute-bound. The sweep measures
  what the model runs, which is the check an isolated micro-benchmark has to pass.
* The small q projection runs at 3.5–3.7× floor at every S. A 1536×1536 bf16 GEMM
  (4.7 MB) is too small to reach the bus's stride.
