# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260924_E11f_pi05_pipelining`  
Written by `bench/pipelining.py` at 2026-09-24T23:15:05Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 2425.369 | 2422.403 | 2428.392 | 2430.291 | 25 | 60.6348 | 0.4123 |
| ae | ae | eager | 237.743 | 237.216 | 238.924 | 247.100 | 253 | 60.1968 | 4.2029 |
| full | full | eager | 2649.448 | 2646.322 | 2658.313 | 2660.517 | 23 | 60.9713 | 0.3772 |
| prefix | prefix | reduce-overhead | 653.661 | 651.233 | 657.429 | 662.758 | 92 | 60.1693 | 1.529 |
| ae | ae | reduce-overhead | 242.196 | 241.712 | 243.294 | 249.988 | 248 | 60.1039 | 4.1262 |
| full | full | reduce-overhead | 708.057 | 705.586 | 711.815 | 714.670 | 85 | 60.2106 | 1.4117 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 339.232 | 238.2996 | 1.4236 | 54.082 | 0.8406 | 184.217 | 0.7730 |
| eager | 10 | 2 | 20 | 843.538 | 596.0173 | 1.4153 | 135.385 | 0.8395 | 460.632 | 0.7729 |
| reduce-overhead | 4 |  |  |  | 244.4213 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 611.2326 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 252.712 | 253.186 | 253.423 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 2410.099 | 2413.444 | 2413.786 | 30 | 9.5369 |
| host_sequence | with_prefix_loop | two_processes | eager | 254.171 | 268.036 | 268.196 | 30 | 1.0058 |
| host_sequence | alone | none | reduce-overhead | 253.432 | 253.959 | 254.352 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 696.911 | 701.123 | 703.113 | 30 | 2.7499 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 253.753 | 254.409 | 254.550 | 30 | 1.0013 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260924_E11f_pi05_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
