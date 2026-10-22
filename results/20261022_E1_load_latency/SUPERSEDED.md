Superseded by `results/20261022_E1_load_latency_v2/` (T003 review): every repeat restarted at the same line and walked
200,000 lines = 25.6 MB, which fits the 32 MiB L2, so repeats beyond the first partly hit L2 and the "DRAM" rows
(64 MiB–1 GiB, 387–549 cycles) are underestimates. v2 flushes the L2 before each repeat, starts at a random line and
walks at most one cycle: DRAM 817–832 cycles. The L1 / L2 rows (≤ 24 MiB) are unaffected and agree with v2.
