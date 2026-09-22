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
| copy | 119.0 | 12.3 % |
| concatenate (KV cache) | 78.0 | 8.0 % |
| reduction / normalisation | 75.0 | 7.7 % |
| other / softmax / memset | 63.0 | 6.5 % |

The expert's GEMMs are **17 %** of the launches in a step; the rest is elementwise work, copies,
KV concatenation and normalisation. The 37 `reduce_kernel` + 37 `rsqrt` per step confirm 36
RMSNorm invocations per step — 18 layers × 2 — measured rather than assumed.

Traces are committed compressed (`zstd -19`, ~23×); `raw/ORIGINALS.json` records each
original's sha256 and size.
