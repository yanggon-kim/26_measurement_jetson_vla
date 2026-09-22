# E2-aux — kernel launch count for one π0 inference at N = 10 (T005)

`torch.profiler` (CUDA activity needs root on this device) over one inference, **counts only** —
the profiler perturbs times, so no latency is taken from here. This directory is one point of a
slope; the analysis that uses it is in `../20260922_E2_kernel_counts/`.

`kernels_per_step.csv`'s `launches_over_N_not_per_step` column is total ÷ N and is **not** a
per-step figure — vision and the VLM prefix run once per inference. The per-step number is the
slope between N = 2 and N = 10: **969 kernel launches per denoising step**, plus 2,629 fixed per
inference.

The chrome trace is committed compressed (`zstd -19`, ~23×); `raw/ORIGINALS.json` records the
original's sha256, size and secrets scan (run before compression).
