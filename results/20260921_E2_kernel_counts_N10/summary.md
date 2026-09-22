# E2-aux — kernel launch count for one π0 inference at N = 10 (T005)

`torch.profiler` (CUDA activity needs root on this device) over one inference, counts only —
the profiler perturbs times, so no latency is taken from here. This directory is one point of
a two-point slope; the analysis that uses it lives in
`../20260921_E2_kernel_counts/` (`kernels_per_step_by_name.csv`,
`kernels_per_step_by_category.csv`, `summary.md`).

| N | total kernel launches | distinct kernel names |
|---|---|---|
| 10 | 12,320 | 82 |

Combined with the other point, the slope gives **969 kernel launches per denoising step** and
2,629 fixed per inference (vision + VLM prefix + post-processing).

The chrome trace is committed compressed (`zstd -19`, ~23×); `raw/ORIGINALS.json` records the
original's sha256 and size.
