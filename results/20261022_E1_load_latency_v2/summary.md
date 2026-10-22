# E1 step 4: dependent-load latency, v2 (T003, optional; supersedes `results/20261022_E1_load_latency/`) — 2026-10-22, MAXN, gate passed

One thread, random single cycle over 128-B-spaced indices. ≤ L2: 200,000 hops after a warm pass. > L2: L2 flushed
(256 MiB written) before each repeat, random start, ≤ one cycle (no line twice). 5 repeats per size; cycles from `clock64`.

| working set | cycles / load, median (min–max) | ns / load, median | level |
|---|---|---|---|
| 16 KiB | 39.8 (39.8–39.8) | 25.4 | L1 |
| 64 KiB | 40.2 (40.2–40.2) | 25.6 | L1 |
| 256 KiB | 151.4 (151.4–151.5) | 96.3 | L1 → L2 |
| 1 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 4 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 16 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 24 MiB | 255.3 (255.3–283.0) | 162.2 | L2 |
| 64 MiB | 816.6 (800.6–816.8) | 518.6 | DRAM (cold walk) |
| 256 MiB | 822.9 (822.7–825.2) | 522.5 | DRAM (cold walk) |
| 1024 MiB | 832.2 (827.6–834.5) | 528.5 | DRAM (cold walk) |

L1 ≈ 40 cycles (25 ns); L2 ≈ 255 cycles (162 ns); DRAM ≈ 817–832 cycles (≈ 520 ns, includes TLB effects of a
random walk over 64 MiB–1 GiB). Repeats agree within 2 % beyond L2.
