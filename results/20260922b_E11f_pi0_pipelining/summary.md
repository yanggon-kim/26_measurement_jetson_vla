# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260922b_E11f_pi0_pipelining`  
Written by `bench/pipelining.py` at 2026-09-22T15:29:50Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 478.645 | 477.081 | 479.195 | 479.456 | 126 | 60.2851 | 2.0901 |
| ae | ae | eager | 221.127 | 220.653 | 221.937 | 222.876 | 272 | 60.1676 | 4.5207 |
| full | full | eager | 704.128 | 703.437 | 704.869 | 705.047 | 86 | 60.5556 | 1.4202 |
| prefix | prefix | reduce-overhead | 340.389 | 340.189 | 340.612 | 340.962 | 177 | 60.2502 | 2.9377 |
| ae | ae | reduce-overhead | 221.809 | 221.338 | 222.425 | 223.344 | 271 | 60.1207 | 4.5076 |
| full | full | reduce-overhead | 363.201 | 362.827 | 363.811 | 365.991 | 166 | 60.3083 | 2.7525 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 339.106 | 221.8142 | 1.5288 | 54.145 | 0.8403 | 167.669 | 0.7559 |
| eager | 10 | 2 | 20 | 857.764 | 554.0465 | 1.5482 | 135.281 | 0.8423 | 418.766 | 0.7558 |
| reduce-overhead | 4 |  |  |  | 223.7148 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 557.6061 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 249.081 | 249.299 | 249.554 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 559.243 | 559.494 | 559.505 | 30 | 2.2452 |
| host_sequence | with_prefix_loop | two_processes | eager | 249.476 | 250.046 | 250.226 | 30 | 1.0016 |
| host_sequence | alone | none | reduce-overhead | 249.200 | 249.365 | 249.394 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 421.289 | 421.580 | 421.595 | 30 | 1.6906 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 249.739 | 250.047 | 250.139 | 30 | 1.0022 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260922b_E11f_pi0_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
