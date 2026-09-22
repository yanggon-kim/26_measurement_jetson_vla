# E2-aux — how many GPU kernels a denoising step launches (T005)

`torch.profiler` (CUDA activity needs root on this device) over one π0 inference at
N ∈ {2, 4, 10}. Counts only — the profiler perturbs times, so no latency is taken from here.
Per-step figures are the **slope** between N = 2 and N = 10, which separates the expert (once
per step) from vision and the VLM prefix (once per inference).

| N | total kernel launches |
|---|---|
| 2 | 4,567 |
| 4 | 6,505 |
| 10 | 12,320 |

**969 kernel launches per denoising step**, plus 2,629 fixed per inference. The linear fit
predicts N = 4 exactly (6,505 predicted vs 6,505 measured).

| category | per step | share |
|---|---|---|
| elementwise | 466.1 | 48.1 % |
| GEMM (tensor/CUDA core) | 168.0 | 17.3 % |
| copy | 157.0 | 16.2 % |
| concatenate (KV cache) | 78.0 | 8.0 % |
| reduction / normalisation | 75.0 | 7.7 % |
| softmax / other / memset | 25.0 | 2.6 % |

The expert's GEMMs are **17 %** of the launches in a step. The 37.0 `reduce_kernel` + 37.0
`rsqrt` per step are exactly 37, not 36: 36 are the two-per-layer RMSNorms (18 × 2) and one is a
further normalisation — an earlier draft assumed 36 and wrongly said these counts confirmed it.

Traces are committed compressed (`zstd -19`, ~23×) in the per-N directories, with each
original's sha256, size and pre-compression secrets scan in their `raw/ORIGINALS.json`.
