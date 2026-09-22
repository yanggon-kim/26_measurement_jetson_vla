# E11a - pi0 host segments (protocol §13.1, §13.2)

Run directory: `20260922b_E11a_pi0_hostseg`  
Written by `bench/e11_summaries.py` at 2026-09-22T15:29:33Z.

## Answer — per-segment latency, `wall_isolated`, `gap_ms = 0` (the §13.1 primary)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsnorm1` | 0 | 0.3961 | 0.2663 | 0.0463 | 9.0 | 106496 + 104448 | 417792 |
| `rope` | 0 | 0.4618 | 0.1930 | 0.0617 | 10.1 | 287232 + 235008 | 705024 |
| `softmax` | 0 | 0.1038 | 0.1403 | 0.0518 | 2.1 | 2122416 + 2122416 | 1768680 |
| `residual_rmsnorm2` | 0 | 0.4328 | 0.2410 | 0.0499 | 10.1 | 313344 + 208896 | 470016 |
| `gelu_gate_mul_up` | 0 | 0.1072 | 0.1462 | 0.0226 | 2.1 | 835584 + 417792 | 1880064 |
| `partials_residual` | 0 | 0.1144 | 0.1420 | 0.0225 | 2.1 | 522240 + 104448 | 208896 |
| `residual` | 0 | 0.0630 | 0.1446 | 0.0227 | 1.1 | 208896 + 104448 | 52224 |
| `attention_gpu` | 0 | 0.5295 | 0.3870 | 0.3154 | 7.1 | 9433776 + 3038784 | 362225664 |
| `cast_scores_fp32_fp16` | 0 | 0.0685 | 0.1407 | 0.0227 | 1.0 | 1414944 + 707472 | 0 |
| `rmsnorm1` | 9 | 0.3994 | 0.2258 | 0.0458 | 9.0 | 106496 + 104448 | 417792 |
| `rope` | 9 | 0.4624 | 0.1929 | 0.0609 | 10.1 | 287232 + 235008 | 705024 |
| `softmax` | 9 | 0.1042 | 0.1395 | 0.0423 | 2.1 | 2122416 + 2122416 | 1768680 |
| `residual_rmsnorm2` | 9 | 0.4337 | 0.2407 | 0.0498 | 10.1 | 313344 + 208896 | 470016 |
| `gelu_gate_mul_up` | 9 | 0.1072 | 0.1476 | 0.0226 | 2.1 | 835584 + 417792 | 1880064 |
| `partials_residual` | 9 | 0.1143 | 0.1435 | 0.0229 | 2.1 | 522240 + 104448 | 208896 |
| `residual` | 9 | 0.0624 | 0.1458 | 0.0227 | 1.1 | 208896 + 104448 | 52224 |
| `attention_gpu` | 9 | 0.5302 | 0.3889 | 0.3121 | 7.1 | 9433776 + 3038784 | 362225664 |
| `cast_scores_fp32_fp16` | 9 | 0.0689 | 0.1413 | 0.0225 | 1.0 | 1414944 + 707472 | 0 |

**Per-step extras** (`order_in_layer = -1`: once per denoising step, not once per layer)

| segment | layer | eager ms | compiled ms | cudagraph ms | kernels, eager | bytes read + written | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `suffix_embed` | -1 | 1.4466 | 0.8492 | not measured | 27.2 |  |  |
| `final_norm_out_proj` | -1 | 0.5920 | not measured | 0.0795 | 11.0 |  |  |
| `euler_update` | -1 | 0.1036 | 0.1483 | 0.0228 | 2.1 |  |  |

`gap_ms = 2`, `gpu_event` and `wall_launch` for every row, with achieved GB/s and TFLOP/s where the segment has bytes and FLOPs, are in `segment_summary.csv`; the byte and FLOP formulas and every input's shape, dtype and contiguity in `segments.csv`; the 216,000 per-iteration samples in `raw/segment_samples.csv.zst`; kernel counts, their `key_averages` table and one Chrome trace per (segment, layer, style) in `raw/profiler/`.

## A whole layer, six ways

| key | exec_style | layer | median ms | p5 | p95 | n | vs the analytic 0.16 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `whole_layer_graph` | cudagraph | layer=0 | **0.1827** | 0.1805 | 0.1854 | 1000 | 1.14x |
| `sum_of_segments` | cudagraph | layer=0 | **0.2548** |  |  | 6 | 1.59x |
| `sequence` | compiled | layer=0 | **0.4455** | 0.4335 | 0.4728 | 1000 | 2.78x |
| `sum_of_segments` | compiled | layer=0 | **1.1287** |  |  | 6 | 7.05x |
| `sequence` | eager | layer=0 | **1.5290** | 1.5103 | 1.5593 | 1000 | 9.56x |
| `sum_of_segments` | eager | layer=0 | **1.6162** |  |  | 6 | 10.10x |
| `whole_layer_graph` | cudagraph | layer=9 | **0.1763** | 0.1739 | 0.1790 | 1000 | 1.10x |
| `sum_of_segments` | cudagraph | layer=9 | **0.2444** |  |  | 6 | 1.53x |
| `sequence` | compiled | layer=9 | **0.4474** | 0.4360 | 0.4742 | 1000 | 2.80x |
| `sum_of_segments` | compiled | layer=9 | **1.0899** |  |  | 6 | 6.81x |
| `sequence` | eager | layer=9 | **1.5283** | 1.5088 | 1.5557 | 1000 | 9.55x |
| `sum_of_segments` | eager | layer=9 | **1.6211** |  |  | 6 | 10.13x |

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
    exec python bench/host_segments.py --outdir results/20260922b_E11a_pi0_hostseg --mode count'
python bench/host_segments.py --outdir results/20260922b_E11a_pi0_hostseg --warmup 20 --timed 1000 \
    --kernels-csv results/20260922b_E11a_pi0_hostseg/kernels_per_call.csv
bash bench/run_nsys.sh results/20260922b_E11a_pi0_hostseg host_sequences -- \
    python bench/host_segments.py --outdir results/20260922b_E11a_pi0_hostseg --mode nsys
```

## Not measured

- suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture
Search for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cuda/cuda-runtime-
- final_norm_out_proj exec_style=compiled: BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trac
