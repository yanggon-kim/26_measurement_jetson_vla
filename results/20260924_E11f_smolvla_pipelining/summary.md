# E11f — pipelining and overlap (protocol §13.7)

Run directory: `20260924_E11f_smolvla_pipelining`  
Written by `bench/pipelining.py` at 2026-09-25T01:50:34Z.

## Answer — throughput of the three loops

| loop | phase | mode | median ms | p5 | p95 | max | n | wall s | inf/s |
|---|---|---|---|---|---|---|---|---|---|
| prefix | prefix | eager | 98.816 | 98.482 | 100.790 | 128.256 | 605 | 60.0626 | 10.0728 |
| ae | ae | eager | 242.876 | 242.298 | 243.694 | 250.086 | 247 | 60.0238 | 4.115 |
| full | full | eager | 345.690 | 344.795 | 348.579 | 353.797 | 174 | 60.2503 | 2.888 |
| prefix | prefix | reduce-overhead | 96.669 | 96.441 | 97.034 | 101.535 | 621 | 60.0548 | 10.3405 |
| ae | ae | reduce-overhead | 242.902 | 242.474 | 243.891 | 249.299 | 247 | 60.0191 | 4.1154 |
| full | full | reduce-overhead | 57.231 | 57.123 | 57.576 | 58.344 | 1048 | 60.0157 | 17.4621 |

## GPU idle inside the denoising loop

| mode | N | inferences | ae_step spans | ae_total ms (nsys) | ae_total ms (unprofiled) | inflation | kernel busy ms | idle frac (raw) | idle ms (corrected) | **idle frac (corrected)** |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 4 | 2 | 8 | 343.540 | 241.5633 | 1.4222 | 54.058 | 0.8426 | 187.505 | 0.7762 |
| eager | 10 | 2 | 20 | 831.605 | 603.6159 | 1.3777 | 135.282 | 0.8373 | 468.334 | 0.7759 |
| reduce-overhead | 4 |  |  |  | 243.2171 |  |  |  |  |  |
| reduce-overhead | 10 |  |  |  | 607.028 |  |  |  |  |  |

**Quote the corrected column.** Two corrections are folded in:

1. The idle is computed **per inference** and averaged, not as one envelope over the whole capture. An envelope across all NVTX instances counts the gap between inferences — the next inference's vision and VLM prefix, which saturate the GPU — as expert idle.
2. nsys instruments every CUDA API call, and this workload is launch-bound, so the wall time it reports for the region is inflated (see the `inflation` column). Kernel durations are **not** inflated — nsys reads them from the GPU — so `(profiled wall − busy) / profiled wall` charges nsys's own overhead to the model. The corrected figure divides the same measured kernel-busy time by the **unprofiled** wall time of the identical loop, measured in the same process a few seconds earlier with no profiler attached.

## Concurrency — as two streams and as two processes

| activity | condition | sharing | mode | median ms | p95 | max | n | slowdown |
|---|---|---|---|---|---|---|---|---|
| host_sequence | alone | none | eager | 124.203 | 124.374 | 124.470 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | eager | 222.042 | 222.476 | 222.689 | 30 | 1.7877 |
| host_sequence | with_prefix_loop | two_processes | eager | 124.393 | 124.774 | 124.816 | 30 | 1.0015 |
| host_sequence | alone | none | reduce-overhead | 124.276 | 124.382 | 124.451 | 30 |  |
| host_sequence | with_prefix_loop | two_streams | reduce-overhead | 221.210 | 221.683 | 221.890 | 30 | 1.78 |
| host_sequence | with_prefix_loop | two_processes | reduce-overhead | 124.876 | 125.072 | 125.117 | 30 | 1.0048 |

## Conditions

π0, bf16, MAXN + jetson_clocks, N = 4 unless the row says otherwise. Region instrumentation forces a graph break under `torch.compile`, so the NVTX-derived idle fraction is an eager measurement; the compiled rows report end-to-end only.

## Reproduce

```
python bench/pipelining.py --outdir results/20260924_E11f_smolvla_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
