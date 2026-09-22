# E2 — the π0 action expert, one kernel at a time (T005)

MAXN, 16 SMs, GPU pinned 1300.5 MHz. Shapes read from the checkpoint: hidden 1024, intermediate
4096, 18 layers, 8 query heads, **1 KV head** (so k and v are two separate 1024→256 GEMMs, not
one 1024→512 — `docs/01_measurement_plan.md` corrected). Floors use E1's **measured**
187.47 GB/s.

Each operator is timed three ways, because a single call between one CUDA-event pair carries a
fixed cost of its own: `eager_single_ms` (one call — reported, never multiplied out),
`marginal_us` (3-point least-squares fit of `t(k) = fixed + k·marginal` over chains of
k ∈ {1,8,64}, each at 20 warm-ups / 100 timed), and a CUDA-graph replay across all 18 layers
(GPU time). `marginal_us` is `max(CPU dispatch, GPU time)`, so it equals the dispatch cost only
while the call is CPU-bound.

| at S = 51 | marginal/call | GPU/call | marginal ÷ GPU | × floor | % of peak BW |
|---|---|---|---|---|---|
| `gate_proj` / `up_proj` (8.39 MB) | 64 µs | 53 µs | 1.2× | 1.11× | 90 % |
| `down_proj` (8.39 MB) | 62 µs | 55 µs | 1.1× | 1.15× | 87 % |
| `q_proj` / `o_proj` (4.19 MB) | 62 µs | 29–30 µs | 2.0–2.2× | 1.18–1.26× | 80–85 % |
| `k_proj` / `v_proj` (0.52 MB) | 62 µs | 8 µs | 7.7× | 2.26–2.28× | 44 % |

Every projection is launch-bound at S = 51; the MLP ones only mildly. All 126 projections of a
denoising step (622.85 MB of weights = 97.6 % of the 638 MB T002's `ae_step` floor used): floor
**3.322 ms**, CUDA graph **4.411 ms (1.33×)**, eager **8.656 ms (2.61×)**.

Non-GEMM operators at S = 51: `rmsnorm` 275 µs marginal against 17.7 µs GPU, `rope` 195 vs
21 µs. Both are Python composites, and the model's `PiGemmaRMSNorm` does *more* than the version
benchmarked here, so those are lower bounds.

**Odd:** `ops.csv` measures activations, whose working set usually fits the 4 MB L2 — five rows
exceed the DRAM peak, `elementwise_add` reaching 447 GB/s (238 %). Those columns are cache
bandwidth; the DRAM-floor column is named
`dram_floor_ms_NOT_APPLICABLE_IF_L2_RESIDENT` to say so per row.

Also here: `ncu/sysmem_miss_validation.csv` (reads exact; writes zero in every `*_lookup_miss`
counter and recovered instead by `op_write.sum`; ncu's own SM clock −0.9 % to −1.9 % of pinned,
within §3.2's 2 %), `raw/ncu_clock_check.csv`, and
`raw/t007_ncu_window_verification.csv` (the `--ncu-window` target T007 will use). ncu reports
committed compressed (`zstd -19`, 84×); originals' hashes and pre-compression secrets scans in
`ncu/ORIGINALS.json`.
