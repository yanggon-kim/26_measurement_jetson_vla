# E13 — π0's prefix under stolen memory bandwidth (T016 step 1, EV-02)

π0 `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc`, rung `bf16_both` exactly as E7 (checked against its
`meta.json`), N = 10, batch 1, Jetson AGX Orin 64 GB at MAXN + `jetson_clocks` (GPU 1300.5 MHz, EMC 3199 MHz,
16 SMs), 2026-09-26. The prefix is the **unmodified rung prefix on the whole GPU** (vision + VLM prefill,
`corun.prefix_stage`), reference 251.0–252.4 ms in this session (E7: 251.28 ms).

## Pre-registered rows, answered

| # | quantity | measured | band |
|---|---|---|---|
| 1 | σ at ~94 GB/s stolen, full-time load | **~94 GB/s is not reachable with an SM-free load during the prefix** (max 84.9 GB/s). σ = **1.247 at 77.1 GB/s**, **1.278–1.281 at 82.9–84.9 GB/s**; σ rises monotonically with the load, so σ(94) ≥ 1.28 | **> 1.247: throughput claim broken too** (already at 77 GB/s stolen) |
| 2 | σ at ~94 GB/s, 37 % duty (93 ms from frame start) | **1.125** at the SM-free maximum (≈ 83 GB/s while on; 27.4 GB/s frame-averaged) | **1.04–1.247: latency parity at 16 ch lost, throughput claim survives** |
| 3 | 1b vs 1 + f·(σ_full − 1) | f (share of the prefix's DRAM bytes in the window, from 1d) = 0.433 (start) / 0.315 (end): predicted 1.121 / 1.052 / 1.088 vs measured **1.125 / 1.047 / 1.073** — differences +0.004, −0.005, −0.015 | **within ±0.02: the loss is linear in the bytes exposed** |
| 4 | 1c: confine the GPU to a channel subset | **no** on stock Orin without a boot or hardware change; a boot-configuration route is unknown (not attempted: forbidden) | **no** |
| 8 | share of the prefix's excess reads that are weight re-reads (1d) | **0.67** (modelled GEMM excess, weighted by in-model excess); fit residuals 18–55 % | **on the 2/3 line — not distinguishable from "mixed"**; top kernels of each below |

## 1a — σ vs stolen bandwidth (`results/20260926_E13a_pi0_bwsteal_ce+cpu_MAXN`)

**Load (A1 primary arm, SM-free):** DMA copy-engine `cudaMemcpyAsync` pinned-host → device and device →
pinned-host on two streams (1 host core), plus CPU copy workers on 7 cores (`bench/bwload.py`, kind `both`), in
**a separate process** pinned to cores 4–11 — a load thread inside the harness would contend with the launch-heavy
rung prefix for the GIL (a host-side confound). Harness on cores 0–3. **No kernel:** nsys `-t cuda` of the
copy-engine load alone lists zero kernels and only H2D / D2H memcpys (`raw/load_trace_*.csv`); the CPU part makes
no CUDA calls. Shared with the prefix: the LPDDR5 bus and memory controller, the GPU memory hub / SMMU path and the
CUDA driver (copy engine), CPU L3 / DSU (CPU part). Why both kinds: the copy engine alone reached only 58.5 GB/s
during the prefix (A1 allows combining).

| load during prefix (own byte count, GB/s) | 0 | 20.0 | 40.0 | 59.8 | 77.1 | 82.9 | 84.6 | 84.8 | 84.9 (max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| left (178.5 − load) GB/s | 178.5 | 158.5 | 138.5 | **118.7** | **101.4** | **95.6** | **93.9** | **93.7** | **93.6** |
| prefix loaded, median ms (p5–p95) | 252.17 | 266.88 | 281.16 | 296.49 | 313.06 | 320.74 | 321.11 | 321.53 | 321.29 (320.39–322.31) |
| ref before / after, ms | 252.2/252.0 | 252.4/251.1 | 251.4/250.9 | 251.1/251.0 | 251.0/250.9 | 251.0/250.9 | 250.9/251.2 | 251.2/250.9 | 251.2/251.2 |
| **σ** | **1.000** | **1.060** | **1.120** | **1.181** | **1.247** | **1.278** | **1.279** | **1.281** | **1.279** |

n = 30 frames per segment (ref → loaded → ref), 0 non-finite. Six points leave 90–130 GB/s (A1: ≥ 3).
**σ ≈ 1 + 0.0032 × (GB/s stolen)**, straight up to where the load saturates; the prefix loses ~0.32 % per GB/s.
EMC during the loaded segments: 41 % (prefix alone) → 76 % at the maximum; a pure streaming load reaches 89 %,
so **the bus is not saturated** when the prefix already loses 28 % — the coupling is latency / queueing in the
memory system, not a full bus.

*Anomaly, disclosed:* this session's EMC hog calibration was not reproducible (four fits: top point 122.6–181.8 GB/s
at the same 85–89 % EMC; `NOTE.md`). "Left" therefore uses E12b's 178.5 GB/s; the load's own byte count is the
x-axis; a load-byte EMC fit (2.321 × EMC% − 10.3, R² 0.99986) is given in `sigma_derived.csv`.

## 1b — 37 % duty (`results/20260926_E13b_pi0_bwduty_{start,end}_MAXN`)

Same load, on only for 93 ms of each prefix frame, aligned by a frame-start timestamp shared with the load process.

| window | load while on | σ (n = 30, bracketed) |
|---|---:|---:|
| 0–93 ms, SM-free max | ≈ 83 GB/s | **1.125** |
| 0–93 ms, 40 GB/s | ≈ 40 GB/s | 1.047 |
| last 93 ms of the reference frame, max | ≈ 87 GB/s | 1.073 |

The start of the frame (vision + first Gemma layers) carries 43 % of the prefix's bytes in its first 93 ms, the end
31 % (1d timeline) — so the same burst costs more at the start, exactly as the byte share predicts (row 3). The
end window is placed at the unloaded frame's last 93 ms; the loaded frame is longer, so it covers ~158–251 ms of
~269 ms.

## 1c — channel confinement (`results/20260926_E13c_channel_probe`, `findings.md`)

FACIL's per-allocation map is a proposed memory-controller + OS change (MapID in huge-page PTE bits, multiplexers
in the MC frontend); the paper assumes, not measures, Orin's SoC mapping and simulates the PIM; no code is
released. On the device: 16 MC channels (`tegra_mc_utils/num_channel`), one interleaved System RAM range, no
channel attribute on nvmap heaps / carve-outs / the CUDA allocator; a stride probe shows no power-of-two
interleave cliff (hashed channel bits likely, mapping not identified). The address map lives in the boot-time
BCT, so any route is a boot change — not attempted.

## 1d — per-kernel DRAM traffic (`results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN`)

Node-level ncu of the rung prefix, E12b metric set, 2 frames = 2286 launches, **in chunks of 450**: on this device
ncu dies with `LaunchFailed` after exactly 480 profiled kernels per process (both first attempts; kept under
`ncu/failed_480_kernel_limit/`). Ops and GEMM shapes from torch.profiler (vision from a no-graph profile, 831 / 831
launches name-matched).

| per frame | f0 | f1 | E12b (graph-level) |
|---|---:|---:|---:|
| DRAM reads (L2 fills) | 17.91 GB | 18.06 GB | 17.80 GB |
| writes (≤) | 3.86 GB | 3.86 GB | 3.83 GB |
| kernel time (profiled) | 255.1 ms | 257.2 ms | — |

Sum check passes (< 1.5 %). **GEMMs: 651 launches read 15.4–15.5 GB against 8.08 GB of unique inputs — 7.36–7.43 GB
excess**, 98 % of it in four shapes (`gemm_excess_by_shape.csv`):

| GEMM (M × N × K) | where | launches | excess | split (standalone M / N sweeps, `gemm_split.json`) |
|---|---|---:|---:|---|
| 816 × 16384 × 2048 | Gemma MLP gate / up | 36 | 2.74 GB | **weight re-reads** (a = 0.22 per extra M-tile; weights 67 MB read ~2.2× at M = 816, ~1.1× at M ≤ 408, 4.4× at 1632) |
| 256 × 4304 × 1152 | SigLIP MLP up | 81 | 2.45 GB | mostly weight (0.86) |
| 816 × 2048 × 16384 | Gemma MLP down | 18 | 1.72 GB | **activation re-reads** (its 26.7 MB activation exceeds the 4 MB L2) |
| 816 × 2048 × 2048 | Gemma attention projections | 36 | 0.34 GB | activation |

Weighted H1 share **0.67**. The split is a derived decomposition (`bench/e13d_split.py`: non-negative fit of
excess = a·W·(M-tiles − 1) + b·X·(N-tiles − 1) over same-kernel sweep points); its relative RMSE is 0.18–0.55, and
cuBLAS changes kernels along the sweeps, so 0.67 cannot be told apart from the mixed band. The first, unconstrained
fit (`gemm_sweep_fit.json`) gave negative coefficients and is superseded. Non-GEMM kernels (norms, GELU, softmax,
RoPE) read 2.5 GB/frame; their unique input bytes are not decomposed.

## Not measured
σ at ≥ 94 GB/s stolen with an SM-free load (not reachable); the optional green-context secondary arm (row 1 is
already decided by the SM-free points); the channel-confined prefix (not possible without a boot change).
