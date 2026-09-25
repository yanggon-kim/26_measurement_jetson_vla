# E11a - pi0 host segments (protocol §13.1, §13.2)

Run directory: `20260924_E11a_pi05_hostseg`  
Written by `bench/host_segments.py` at 2026-09-24T21:13:09Z.

## Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsnorm1` | 0 | 0.4983 | 0.3699 | 0.1315 | 0.0 | 104448 + 102400 | 409600 |
| `rope` | 0 | 0.4568 | 0.1952 | 0.0603 | 0.0 | 281600 + 230400 | 691200 |
| `softmax` | 0 | 0.0683 | 0.1412 | 0.0402 | 0.0 | 2443200 + 2443200 | 2036000 |
| `residual_rmsnorm2` | 0 | 0.5722 | 0.3790 | 0.1345 | 0.0 | 409600 + 204800 | 512000 |
| `gelu_gate_mul_up` | 0 | 0.1076 | 0.1480 | 0.0308 | 0.0 | 819200 + 409600 | 1843200 |
| `partials_residual` | 0 | 0.1152 | 0.1432 | 0.0227 | 0.0 | 512000 + 102400 | 204800 |
| `residual` | 0 | 0.0959 | 0.1491 | 0.0223 | 0.0 | 307200 + 102400 | 102400 |
| `attention_gpu` | 0 | 0.5759 | 0.5198 | 0.4253 | 0.0 | 10987456 + 3462400 | 416972800 |
| `cast_scores_fp32_fp16` | 0 | 0.0695 | 0.1437 | 0.0230 | 0.0 | 1628800 + 814400 | 0 |
| `rmsnorm1` | 9 | 0.4989 | 0.3646 | 0.1313 | 0.0 | 104448 + 102400 | 409600 |
| `rope` | 9 | 0.4596 | 0.1962 | 0.0600 | 0.0 | 281600 + 230400 | 691200 |
| `softmax` | 9 | 0.0686 | 0.1412 | 0.0400 | 0.0 | 2443200 + 2443200 | 2036000 |
| `residual_rmsnorm2` | 9 | 0.5718 | 0.3679 | 0.1355 | 0.0 | 409600 + 204800 | 512000 |
| `gelu_gate_mul_up` | 9 | 0.1069 | 0.1486 | 0.0307 | 0.0 | 819200 + 409600 | 1843200 |
| `partials_residual` | 9 | 0.1152 | 0.1463 | 0.0227 | 0.0 | 512000 + 102400 | 204800 |
| `residual` | 9 | 0.0964 | 0.1497 | 0.0226 | 0.0 | 307200 + 102400 | 102400 |
| `attention_gpu` | 9 | 0.5768 | 0.5189 | 0.4260 | 0.0 | 10987456 + 3462400 | 416972800 |
| `cast_scores_fp32_fp16` | 9 | 0.0695 | 0.1440 | 0.0228 | 0.0 | 1628800 + 814400 | 0 |

**Per-step extras** (`order_in_layer = -1`: once per denoising step, not once per layer)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `suffix_embed` | -1 | 1.2136 | 0.6571 | not measured | 0.0 |  |  |
| `final_norm_out_proj` | -1 | 0.6499 | not measured | 0.1586 | 0.0 |  |  |
| `euler_update` | -1 | 0.1036 | 0.1492 | 0.0228 | 0.0 |  |  |

`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in `segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype and contiguity in `segments.csv`; the 216,000 per-iteration samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one Chrome trace per (segment, layer, style) in `raw/profiler/`.

## A whole layer, six ways

| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic 0.16 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `whole_layer_graph` | cudagraph | layer=0 | **0.3374** | 0.3345 | 0.3409 | 1000 | 2.11x |
| `sum_of_segments` | cudagraph | layer=0 | **0.4201** |  |  | 6 | 2.63x |
| `sequence` | compiled | layer=0 | **0.6975** | 0.6801 | 0.7331 | 1000 | 4.36x |
| `sum_of_segments` | compiled | layer=0 | **1.3765** |  |  | 6 | 8.60x |
| `sequence` | eager | layer=0 | **1.7388** | 1.7172 | 1.7720 | 1000 | 10.87x |
| `sum_of_segments` | eager | layer=0 | **1.8184** |  |  | 6 | 11.36x |
| `whole_layer_graph` | cudagraph | layer=9 | **0.3355** | 0.3328 | 0.3389 | 1000 | 2.10x |
| `sum_of_segments` | cudagraph | layer=9 | **0.4202** |  |  | 6 | 2.63x |
| `sequence` | compiled | layer=9 | **0.6940** | 0.6769 | 0.7284 | 1000 | 4.34x |
| `sum_of_segments` | compiled | layer=9 | **1.3649** |  |  | 6 | 8.53x |
| `sequence` | eager | layer=9 | **1.7391** | 1.7158 | 1.7685 | 1000 | 10.87x |
| `sum_of_segments` | eager | layer=9 | **1.8211** |  |  | 6 | 11.38x |

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
    exec python bench/host_segments.py --outdir results/20260924_E11a_pi05_hostseg --mode count'
python bench/host_segments.py --outdir results/20260924_E11a_pi05_hostseg --warmup 20 --timed 1000 \
    --kernels-csv results/20260924_E11a_pi05_hostseg/kernels_per_call.csv
bash bench/run_nsys.sh results/20260924_E11a_pi05_hostseg host_sequences -- \
    python bench/host_segments.py --outdir results/20260924_E11a_pi05_hostseg --mode nsys
```

## Not measured

- suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-
- final_norm_out_proj exec_style=compiled: BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trac
