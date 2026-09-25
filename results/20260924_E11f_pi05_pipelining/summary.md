# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260924_E11f_pi05_pipelining`  
Written by `bench/pipelining.py` at 2026-09-25T03:34:36Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 2422.329 | 2420.161 | 2424.268 | 2425.263 | 25 | 60.5572 | 0.4128 |
| ae | ae | eager | 251.419 | 248.391 | 253.613 | 274.795 | 240 | 60.2128 | 3.9859 |
| full | full | eager | 2666.510 | 2657.503 | 2678.335 | 2679.544 | 23 | 61.349 | 0.3749 |
| prefix | prefix | reduce-overhead | 639.425 | 637.300 | 642.451 | 644.581 | 94 | 60.1206 | 1.5635 |
| ae | ae | reduce-overhead | 250.246 | 248.884 | 254.277 | 264.773 | 239 | 60.0569 | 3.9796 |
| full | full | reduce-overhead | 697.463 | 695.154 | 699.978 | 701.704 | 87 | 60.6785 | 1.4338 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 360.222 | 249.7392 | 1.4424 | 128.880 | 0.6422 | 120.859 | 0.4839 |
| eager | 10 | 2 | 20 | 911.287 | 624.2613 | 1.4598 | 322.278 | 0.6463 | 301.984 | 0.4837 |
| reduce-overhead | 4 |  |  |  | 253.3575 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 632.5408 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 254.708 | 255.060 | 255.140 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 2408.949 | 2411.826 | 2415.997 | 30 | 9.4577 |
| host_sequence | with_prefix_loop | two_processes | eager | 255.685 | 262.481 | 273.494 | 30 | 1.0038 |
| host_sequence | alone | none | reduce-overhead | 254.856 | 255.018 | 255.057 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 645.942 | 648.259 | 648.348 | 30 | 2.5345 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 254.813 | 255.589 | 255.648 | 30 | 0.9998 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260924_E11f_pi05_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
