# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260924_E11f_pi0_pipelining`  
Written by `bench/pipelining.py` at 2026-09-24T21:05:07Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 477.600 | 477.326 | 478.907 | 482.803 | 126 | 60.2021 | 2.0929 |
| ae | ae | eager | 229.035 | 227.589 | 232.528 | 238.097 | 262 | 60.1403 | 4.3565 |
| full | full | eager | 709.778 | 709.012 | 710.660 | 718.002 | 85 | 60.3447 | 1.4086 |
| prefix | prefix | reduce-overhead | 339.417 | 339.171 | 339.941 | 340.282 | 177 | 60.087 | 2.9457 |
| ae | ae | reduce-overhead | 227.902 | 227.304 | 229.021 | 231.705 | 264 | 60.2005 | 4.3853 |
| full | full | reduce-overhead | 364.260 | 363.581 | 365.486 | 366.968 | 165 | 60.1182 | 2.7446 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 343.186 | 228.5123 | 1.5018 | 54.147 | 0.8422 | 174.365 | 0.7630 |
| eager | 10 | 2 | 20 | 865.168 | 571.6329 | 1.5135 | 135.354 | 0.8436 | 436.279 | 0.7632 |
| reduce-overhead | 4 |  |  |  | 229.1316 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 571.8413 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 250.084 | 250.372 | 250.584 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 560.611 | 560.909 | 560.951 | 30 | 2.2417 |
| host_sequence | with_prefix_loop | two_processes | eager | 250.536 | 251.175 | 251.892 | 30 | 1.0018 |
| host_sequence | alone | none | reduce-overhead | 250.263 | 250.550 | 250.766 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 423.151 | 423.507 | 423.600 | 30 | 1.6908 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 250.291 | 250.595 | 250.598 | 30 | 1.0001 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260924_E11f_pi0_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
