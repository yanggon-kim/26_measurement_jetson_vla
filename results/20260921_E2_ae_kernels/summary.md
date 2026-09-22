# E2 — the π0 action expert, one kernel at a time (T005)

MAXN, 16 SMs, GPU pinned 1300.5 MHz. Shapes read from the checkpoint, not a table: hidden 1024,
intermediate 4096, 18 layers, 8 query heads, **1 KV head** (so k and v are two separate
1024→256 GEMMs, not one 1024→512 — `docs/01_measurement_plan.md` corrected). Floors use E1's
**measured** 185.59 GB/s, not the spec sheet.

Each operator is timed three ways, because a single call between one CUDA-event pair carries a
fixed ~45 µs of timed-region cost that is not the call: `eager_single_ms` (one call, reported
but never multiplied out), `eager_chain_ms` (marginal cost in a 64-call chain — what the model
actually pays), and a CUDA-graph replay over all 18 layers (GPU time only).

| at S = 51 | CPU/call | GPU/call | × memory floor | % of peak BW |
|---|---|---|---|---|
| `gate_proj` / `up_proj` (8.39 MB) | 64 µs | 53 µs | 1.11× | 90 % |
| `down_proj` (8.39 MB) | 62 µs | 55 µs | 1.15× | 87 % |
| `q_proj` / `o_proj` (4.19 MB) | 62 µs | 29–31 µs | 1.18–1.27× | 79–85 % |
| `k_proj` / `v_proj` (0.52 MB) | 62 µs | 8 µs | 2.28× | 44 % |

**CPU dispatch is ~62 µs per projection whatever the shape**; GPU time varies 7× with weight
size. Every projection is launch-bound at S = 51.

All 126 projections of a denoising step (622.85 MB of weights = 97.6 % of the 638 MB T002's
`ae_step` floor used): floor **3.356 ms**, CUDA graph **4.411 ms (1.31×)**, eager
**8.674 ms (2.58×)**. So the expert's weight streaming is near-optimal; the 15.3× gap of a whole
eager step is elsewhere.

Non-GEMM operators at S = 51: `rmsnorm` 282 µs CPU against 17.6 µs GPU (16×), `rope` 199 vs
21 µs. Both are Python composites, and the model's `PiGemmaRMSNorm` does *more* than the version
benchmarked here, so those are lower bounds.

**Odd:** `ops.csv` measures activations, whose working set fits the 4 MB L2 — `l2_GBps` reaches
447 GB/s, 2.4× the DRAM peak. Those columns are cache bandwidth and must not be compared with a
DRAM roofline; the CSV deliberately carries no DRAM-roofline column for them.

Also here: `ncu/sysmem_miss_validation.csv` (the DRAM-traffic substitute — reads exact, writes
zero in every `*_lookup_miss` counter and recovered instead by `op_write.sum`) and
`raw/ncu_clock_check.csv` (protocol §3.2, per profiled run). ncu reports committed compressed
(`zstd -19`, 32–114×) with originals' hashes in `ncu/ORIGINALS.json`.
