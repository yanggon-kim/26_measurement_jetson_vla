# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260924_E11f_smolvla_pipelining`  
Written by `bench/pipelining.py` at 2026-09-25T03:48:57Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 96.499 | 96.230 | 97.021 | 101.110 | 622 | 60.0545 | 10.3573 |
| ae | ae | eager | 237.006 | 236.486 | 237.653 | 238.191 | 254 | 60.2014 | 4.2192 |
| full | full | eager | 339.629 | 338.949 | 340.480 | 341.241 | 177 | 60.1227 | 2.944 |
| prefix | prefix | reduce-overhead | 95.468 | 95.170 | 95.998 | 105.548 | 629 | 60.0922 | 10.4672 |
| ae | ae | reduce-overhead | 238.346 | 237.323 | 241.441 | 246.962 | 252 | 60.2078 | 4.1855 |
| full | full | reduce-overhead | 57.242 | 57.067 | 58.127 | 60.561 | 1047 | 60.046 | 17.4366 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 369.814 | 236.9444 | 1.5608 | 27.163 | 0.9266 | 209.782 | 0.8854 |
| eager | 10 | 2 | 20 | 924.636 | 592.404 | 1.5608 | 67.646 | 0.9268 | 524.758 | 0.8858 |
| reduce-overhead | 4 |  |  |  | 239.8515 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 600.2418 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 124.010 | 124.160 | 124.186 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 220.652 | 221.085 | 221.265 | 30 | 1.7793 |
| host_sequence | with_prefix_loop | two_processes | eager | 124.073 | 124.203 | 124.236 | 30 | 1.0005 |
| host_sequence | alone | none | reduce-overhead | 124.162 | 124.298 | 124.336 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 220.161 | 221.053 | 221.342 | 30 | 1.7732 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 124.193 | 124.487 | 124.584 | 30 | 1.0002 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260924_E11f_smolvla_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
