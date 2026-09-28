# E11a - pi0 host segments (protocol §13.1, §13.2)

Run directory: `20260928_E11a_pi05_hostseg`  
Written by `bench/host_segments.py` at 2026-09-28T21:18:00Z.

## Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsnorm1` | 0 | 0.4967 | 0.3681 | 0.1311 | 9.1 | 104448 + 102400 | 409600 |
| `rope` | 0 | 0.4539 | 0.1931 | 0.0588 | 10.1 | 281600 + 230400 | 691200 |
| `softmax` | 0 | 0.0667 | 0.1382 | 0.0387 | 1.1 | 2443200 + 2443200 | 2036000 |
| `residual_rmsnorm2` | 0 | 0.5671 | 0.3654 | 0.1349 | 11.1 | 409600 + 204800 | 512000 |
| `gelu_gate_mul_up` | 0 | 0.1057 | 0.1448 | 0.0340 | 2.1 | 819200 + 409600 | 1843200 |
| `partials_residual` | 0 | 0.1134 | 0.1411 | 0.0222 | 2.1 | 512000 + 102400 | 204800 |
| `residual` | 0 | 0.0945 | 0.1473 | 0.0221 | 2.1 | 307200 + 102400 | 102400 |
| `attention_gpu` | 0 | 0.5753 | 0.5190 | 0.4251 | 6.1 | 10987456 + 3462400 | 416972800 |
| `cast_scores_fp32_fp16` | 0 | 0.0691 | 0.1409 | 0.0218 | 1.0 | 1628800 + 814400 | 0 |
| `rmsnorm1` | 9 | 0.4977 | 0.3614 | 0.1308 | 9.1 | 104448 + 102400 | 409600 |
| `rope` | 9 | 0.4578 | 0.1929 | 0.0585 | 10.1 | 281600 + 230400 | 691200 |
| `softmax` | 9 | 0.0673 | 0.1392 | 0.0383 | 1.1 | 2443200 + 2443200 | 2036000 |
| `residual_rmsnorm2` | 9 | 0.5696 | 0.3660 | 0.1359 | 11.1 | 409600 + 204800 | 512000 |
| `gelu_gate_mul_up` | 9 | 0.1051 | 0.1460 | 0.0329 | 2.1 | 819200 + 409600 | 1843200 |
| `partials_residual` | 9 | 0.1131 | 0.1422 | 0.0224 | 2.1 | 512000 + 102400 | 204800 |
| `residual` | 9 | 0.0944 | 0.1491 | 0.0220 | 2.1 | 307200 + 102400 | 102400 |
| `attention_gpu` | 9 | 0.5751 | 0.5181 | 0.4260 | 6.1 | 10987456 + 3462400 | 416972800 |
| `cast_scores_fp32_fp16` | 9 | 0.0684 | 0.1413 | 0.0220 | 1.0 | 1628800 + 814400 | 0 |

**Per-step extras** (`order_in_layer = -1`: once per denoising step, not once per layer)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `suffix_embed` | -1 | 1.2075 | 0.6495 | not measured | 22.0 |  |  |
| `final_norm_out_proj` | -1 | 0.6455 | not measured | 0.1585 | 10.1 |  |  |
| `euler_update` | -1 | 0.1013 | 0.1475 | 0.0223 | 2.1 |  |  |

`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in `segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype and contiguity in `segments.csv`; the 216,000 per-iteration samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one Chrome trace per (segment, layer, style) in `raw/profiler/`.

## A whole layer, six ways

| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic 0.16 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `whole_layer_graph` | cudagraph | layer=0 | **0.3344** | 0.3317 | 0.3381 | 1000 | 2.09x |
| `sum_of_segments` | cudagraph | layer=0 | **0.4197** |  |  | 6 | 2.62x |
| `sequence` | compiled | layer=0 | **0.6936** | 0.6748 | 0.7369 | 1000 | 4.34x |
| `sum_of_segments` | compiled | layer=0 | **1.3506** |  |  | 6 | 8.44x |
| `sequence` | eager | layer=0 | **1.7309** | 1.7058 | 1.7703 | 1000 | 10.82x |
| `sum_of_segments` | eager | layer=0 | **1.8035** |  |  | 6 | 11.27x |
| `whole_layer_graph` | cudagraph | layer=9 | **0.3335** | 0.3310 | 0.3367 | 1000 | 2.08x |
| `sum_of_segments` | cudagraph | layer=9 | **0.4188** |  |  | 6 | 2.62x |
| `sequence` | compiled | layer=9 | **0.6874** | 0.6700 | 0.7343 | 1000 | 4.30x |
| `sum_of_segments` | compiled | layer=9 | **1.3478** |  |  | 6 | 8.42x |
| `sequence` | eager | layer=9 | **1.7287** | 1.7067 | 1.7612 | 1000 | 10.80x |
| `sum_of_segments` | eager | layer=9 | **1.8106** |  |  | 6 | 11.32x |

`sum_of_segments` is derived (the six mapping segments added up); `sequence` and `whole_layer_graph` are measured directly. The ordering is the finding: **one CUDA graph over the whole layer is the only implementation that comes near the mapping's 0.16 ms**, and it beats a compiled sequence by about 2.4x. Compiling the sequence is worth ~2.5x over compiling the six segments separately — fusion does cross the segment boundaries the mapping treats as separable — but it lands ~2.8x above 0.16 ms, not below it.

## Conditions

pi0 @ `45dcc8fc`, bf16, MAXN + jetson_clocks, P = 816 processed (533 valid), S = 51, N = 4. Layers 0 and 9 of the action expert. >= 20 warm-ups and 1,000 timed calls per (segment, style, timing) — §13.1 asks for >= 20 and >= 1,000. Inputs are the model's own tensors, captured with hooks from one real denoising step of the fixture and replayed through the model's own callables; the prefix K/V are read from the cache **before** the layer updates it, so the replayed attention runs over the real T = P + S = 867 keys rather than S = 51.

## The caveat that governs every compiled number

A compiled callable whose outputs are discarded is dead-code-eliminated. Measured on this device to size the error: a compiled 2048^3 bf16 matmul costs **0.0662 ms** when its result is discarded and **0.6605 ms** when it is returned. An earlier version of this experiment returned `None` from the timed sequence and reported it at 0.085 ms, which measured nothing. Every timed callable here returns its results.

## Reproduce

```
# kernel counts and profiler traces need root: CUPTI does, on this device.
# They run as a SEPARATE pass because attaching a profiler inflates every
# subsequent call by ~30 %.
sudo -n /bin/sh -c 'cd $PWD && HOME=$HOME HF_HUB_OFFLINE=1 \
    exec python bench/host_segments.py --outdir results/20260928_E11a_pi05_hostseg --mode count'
python bench/host_segments.py --outdir results/20260928_E11a_pi05_hostseg --warmup 20 --timed 1000 \
    --kernels-csv results/20260928_E11a_pi05_hostseg/kernels_per_call.csv
bash bench/run_nsys.sh results/20260928_E11a_pi05_hostseg host_sequences -- \
    python bench/host_segments.py --outdir results/20260928_E11a_pi05_hostseg --mode nsys
```

## Not measured

- suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-
- final_norm_out_proj exec_style=compiled: BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trac
