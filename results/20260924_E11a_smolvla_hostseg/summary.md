# E11a - pi0 host segments (protocol §13.1, §13.2)

Run directory: `20260924_E11a_smolvla_hostseg`  
Written by `bench/host_segments.py` at 2026-09-24T23:17:58Z.

## Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsnorm1` | 0 | 0.2663 | 0.2292 | 0.0378 | 0.0 | 73440 + 72000 | 288000 |
| `residual_rmsnorm2` | 0 | 0.4033 | 0.2264 | 0.0477 | 0.0 | 216000 + 144000 | 324000 |
| `gelu_gate_mul_up` | 0 | 0.1084 | 0.1443 | 0.0238 | 0.0 | 409600 + 204800 | 921600 |
| `residual` | 0 | 0.0921 | 0.1443 | 0.0238 | 0.0 | 144000 + 72000 | 36000 |
| `attention_whole` | 0 | 0.0112 | 0.0236 | 0.0178 | 0.0 | 0 + 96000 | 0 |
| `rmsnorm1` | 9 | 0.3380 | 0.2290 | 0.0422 | 0.0 | 73440 + 72000 | 288000 |
| `residual_rmsnorm2` | 9 | 0.4047 | 0.2268 | 0.0432 | 0.0 | 216000 + 144000 | 324000 |
| `gelu_gate_mul_up` | 9 | 0.1069 | 0.1476 | 0.0235 | 0.0 | 409600 + 204800 | 921600 |
| `residual` | 9 | 0.0927 | 0.1457 | 0.0235 | 0.0 | 144000 + 72000 | 36000 |
| `attention_whole` | 9 | 0.0112 | 0.0235 | 0.0174 | 0.0 | 0 + 96000 | 0 |

**Per-step extras** (`order_in_layer = -1`: once per denoising step, not once per layer)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `suffix_embed` | -1 | 1.2732 | 0.7142 | not measured | 0.0 |  |  |
| `final_norm_out_proj` | -1 | 0.5295 | not measured | 0.0672 | 0.0 |  |  |
| `euler_update` | -1 | 0.1037 | 0.1484 | 0.0237 | 0.0 |  |  |

`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in `segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype and contiguity in `segments.csv`; the 216,000 per-iteration samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one Chrome trace per (segment, layer, style) in `raw/profiler/`.

## A whole layer, six ways

| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic 0.16 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `whole_layer_graph` | cudagraph | layer=0 | **0.0674** | 0.0642 | 0.0716 | 1000 | 0.42x |
| `sum_of_segments` | cudagraph | layer=0 | **0.1094** |  |  | 6 | 0.68x |
| `sequence` | compiled | layer=0 | **0.3134** | 0.3070 | 0.3388 | 1000 | 1.96x |
| `sum_of_segments` | compiled | layer=0 | **0.5999** |  |  | 6 | 3.75x |
| `sequence` | eager | layer=0 | **0.7309** | 0.7168 | 0.7584 | 1000 | 4.57x |
| `sum_of_segments` | eager | layer=0 | **0.7780** |  |  | 6 | 4.86x |
| `whole_layer_graph` | cudagraph | layer=9 | **0.0667** | 0.0636 | 0.0707 | 1000 | 0.42x |
| `sum_of_segments` | cudagraph | layer=9 | **0.1089** |  |  | 6 | 0.68x |
| `sequence` | compiled | layer=9 | **0.3128** | 0.3061 | 0.3298 | 1000 | 1.96x |
| `sum_of_segments` | compiled | layer=9 | **0.6034** |  |  | 6 | 3.77x |
| `sequence` | eager | layer=9 | **0.7921** | 0.7769 | 0.8138 | 1000 | 4.95x |
| `sum_of_segments` | eager | layer=9 | **0.8496** |  |  | 6 | 5.31x |

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
    exec python bench/host_segments.py --outdir results/20260924_E11a_smolvla_hostseg --mode count'
python bench/host_segments.py --outdir results/20260924_E11a_smolvla_hostseg --warmup 20 --timed 1000 \
    --kernels-csv results/20260924_E11a_smolvla_hostseg/kernels_per_call.csv
bash bench/run_nsys.sh results/20260924_E11a_smolvla_hostseg host_sequences -- \
    python bench/host_segments.py --outdir results/20260924_E11a_smolvla_hostseg --mode nsys
```

## Not measured

- suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-
- final_norm_out_proj exec_style=compiled: BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trac
