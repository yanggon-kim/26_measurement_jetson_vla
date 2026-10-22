# E2 π0 action-expert kernels, v2 — FAST memory-path state (T003) — 2026-10-22, MAXN, gate passed, 0 OC events

Same code as v3, but it ran immediately after 3.5 min of back-to-back GEMM probes
(`results/20261022_E1_peak_backtoback/`), in the fast memory-path state (`results/20261022_E1_state_probe/`). At
S = 51, gate / up 34.1 / 33.9 µs (v3: 49.6 / 50.0), down 33.1 (36.4), q 18.0 (20.1), o 17.7 (18.5); k / v unchanged
(4.1). The per-step sum is 2.61 ms (v3: 3.29). Kept as evidence of the state, not as the baseline.
