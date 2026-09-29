# E12d arm (c) — the eager co-run as two processes with slim weights, under MPS

* **What:** identical to `…_E12d_pi0_eager_twoproc_N10_MAXN` but with the MPS control daemon running
  (`nvidia-cuda-mps-control -d` as the user; both processes attached as clients of one MPS server — `raw/mps_control.log`,
  8 NEW CLIENT lines over the run; daemon stopped afterwards, no mps-server process left). Bracketed, n = 120, 71 s loop.
* **Equivalence:** bitwise identical to the single-process serial reference on 6 frames; stale control 5/5; 0 non-finite.

| ms (median, p5–p95) | value |
|---|---:|
| A_iso prefix process | 473.85 (473.69–474.36) |
| B_iso expert process | 579.92 (579.22–580.68) |
| T_serial, lockstep | 1056.38 (1055.32–1057.54) |
| **T_pipe (71 s)** | **835.04** |
| A_co / B_co | 509.61 (508.38–511.21) / 837.12 (834.79–839.50) |
| O / φ / p / s | 1.440 / 0.465 / 1.075 / 1.443 |

* **Reading:** MPS **overlaps less** than two plain contexts (T_pipe 835 vs 687 ms, φ 0.46 vs 0.75) and shifts the cost
  onto the expert (s 1.44 vs 1.02). *Interpretation, not measured (no timeline under MPS):* under one MPS server the prefix's large grids and the expert's many small kernels share
  the SMs concurrently, and the host-paced expert loses more than it gains. Without MPS the two contexts presumably time-slice and the
  expert's sparse kernels fit into the prefix's schedule better. Isolated times are ~2 % higher under MPS (B 579.9 vs 568.4).
