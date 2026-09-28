# E11a - pi0 host segments (protocol §13.1, §13.2)

Run directory: `20260928_E11a_smolvla_hostseg`  
Written by `bench/host_segments.py` at 2026-09-28T22:30:01Z.

## Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsnorm1` | 0 | 0.2585 | 0.2267 | 0.0371 | 6.1 | 73440 + 72000 | 288000 |
| `residual_rmsnorm2` | 0 | 0.3904 | 0.2225 | 0.0483 | 10.0 | 216000 + 144000 | 324000 |
| `gelu_gate_mul_up` | 0 | 0.1044 | 0.1430 | 0.0232 | 2.1 | 409600 + 204800 | 921600 |
| `residual` | 0 | 0.0888 | 0.1413 | 0.0232 | 2.0 | 144000 + 72000 | 36000 |
| `attention_whole` | 0 | 3.5091 | 0.9396 | 0.2606 | 63.1 | 0 + 96000 | 0 |
| `rmsnorm1` | 9 | 0.3282 | 0.2246 | 0.0403 | 8.0 | 73440 + 72000 | 288000 |
| `residual_rmsnorm2` | 9 | 0.3889 | 0.2227 | 0.0419 | 10.0 | 216000 + 144000 | 324000 |
| `gelu_gate_mul_up` | 9 | 0.1042 | 0.1438 | 0.0232 | 2.1 | 409600 + 204800 | 921600 |
| `residual` | 9 | 0.0874 | 0.1421 | 0.0232 | 2.0 | 144000 + 72000 | 36000 |
| `attention_whole` | 9 | 2.6054 | 0.9729 | 0.2773 | 43.0 | 0 + 96000 | 0 |

**Per-step extras** (`order_in_layer = -1`: once per denoising step, not once per layer)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `suffix_embed` | -1 | 1.2503 | 0.7009 | not measured | 23.1 |  |  |
| `final_norm_out_proj` | -1 | 0.5202 | not measured | 0.0660 | 10.0 |  |  |
| `euler_update` | -1 | 0.0993 | 0.1461 | 0.0237 | 2.1 |  |  |

`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in `segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype and contiguity in `segments.csv`; the 216,000 per-iteration samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one Chrome trace per (segment, layer, style) in `raw/profiler/`.

## A whole layer, six ways

| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic 0.16 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `whole_layer_graph` | cudagraph | layer=0 | **0.0664** | 0.0628 | 0.0705 | 1000 | 0.41x |
| `sum_of_segments` | cudagraph | layer=0 | **0.1085** |  |  | 6 | 0.68x |
| `sequence` | compiled | layer=0 | **0.3049** | 0.2991 | 0.3297 | 1000 | 1.91x |
| `sum_of_segments` | compiled | layer=0 | **0.5922** |  |  | 6 | 3.70x |
| `sequence` | eager | layer=0 | **0.7110** | 0.6967 | 0.7391 | 1000 | 4.44x |
| `sum_of_segments` | eager | layer=0 | **0.7533** |  |  | 6 | 4.71x |
| `whole_layer_graph` | cudagraph | layer=9 | **0.0655** | 0.0622 | 0.0697 | 1000 | 0.41x |
| `sum_of_segments` | cudagraph | layer=9 | **0.1053** |  |  | 6 | 0.66x |
| `sequence` | compiled | layer=9 | **0.3058** | 0.3002 | 0.3259 | 1000 | 1.91x |
| `sum_of_segments` | compiled | layer=9 | **0.5911** |  |  | 6 | 3.69x |
| `sequence` | eager | layer=9 | **0.7724** | 0.7573 | 0.7955 | 1000 | 4.83x |
| `sum_of_segments` | eager | layer=9 | **0.8213** |  |  | 6 | 5.13x |

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
    exec python bench/host_segments.py --outdir results/20260928_E11a_smolvla_hostseg --mode count'
python bench/host_segments.py --outdir results/20260928_E11a_smolvla_hostseg --warmup 20 --timed 1000 \
    --kernels-csv results/20260928_E11a_smolvla_hostseg/kernels_per_call.csv
bash bench/run_nsys.sh results/20260928_E11a_smolvla_hostseg host_sequences -- \
    python bench/host_segments.py --outdir results/20260928_E11a_smolvla_hostseg --mode nsys
```

## Not measured

- suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-
- final_norm_out_proj exec_style=compiled: BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trac
