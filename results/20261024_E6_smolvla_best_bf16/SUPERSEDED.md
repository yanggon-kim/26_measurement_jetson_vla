# Superseded

Captured before the profiling runs reused the timed run's Inductor / Triton caches. Autotuning chose different expert
kernels in the ncu process than in the nsys process (257 of 284 expert-step kernels differ by name), so the
ncu ↔ nsys join of `ae_step[0]` is invalid. The vision capture also crashed: ncu followed an Inductor compile worker
(fixed with `--target-processes application-only`). Superseded by the `_v2` directories of the same date, which load
the confirmation run's compiled kernels.
