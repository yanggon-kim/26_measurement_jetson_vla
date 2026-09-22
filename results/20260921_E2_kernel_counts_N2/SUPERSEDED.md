# SUPERSEDED by `results/20260922_E2_kernel_counts_N2/`

This run is kept because it is a real measurement, and because the report cites the E1 runs as
evidence of run-to-run bandwidth variation — not because it is the authoritative result.

Superseded on 2026-09-22 after an independent adversarial review (protocol §11) found that it
violated protocol §2 (the chained measurements used 10 warm-ups / 30 timed calls, below the
binding >= 20 / >= 100), and that several derived columns were misnamed or unusable:
`cpu_dispatch_us` is really `max(CPU dispatch, GPU time)` and reads 208 us at S=256;
`nsys_kernel_ms` was in fact ncu's own Duration, which protocol §3.3(3) forbids deriving GB/s
from; and `Memcpy` kernels were classified as "other" rather than as copies, under-counting
copies by 38 launches per step.

**Use `results/20260922_E2_kernel_counts_N2/` instead.** Report: `reports/E1_E2_microbench.md`.
