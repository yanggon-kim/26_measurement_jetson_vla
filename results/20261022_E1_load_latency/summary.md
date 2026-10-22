# E1 step 4: dependent-load latency (T003, optional) — 2026-10-22, MAXN, gate passed

One thread, random single cycle over 128-B-spaced indices, 200,000 dependent loads × 5 repeats per size;
cycles from `clock64`, ns from CUDA events.

| working set | cycles / load, median (min–max) | ns / load, median | level |
|---|---|---|---|
| 16 KiB | 39.8 (39.8–39.8) | 25.4 | L1 |
| 64 KiB | 40.2 (40.2–40.2) | 25.6 | L1 |
| 256 KiB | 151.3 (145.3–151.5) | 96.2 | L1 → L2 transition |
| 1 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 4 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 16 MiB | 255.3 (255.3–255.3) | 162.2 | L2 |
| 24 MiB | 255.3 (255.3–284.7) | 162.2 | L2 |
| 64 MiB | 485.6 (481.8–636.0) | 308.4 | DRAM |
| 256 MiB | 548.8 (540.4–746.3) | 348.5 | DRAM (+TLB) |
| 1024 MiB | 386.8 (383.5–775.1) | 245.7 | DRAM |

L1 hit ≈ 40 cycles, L2 hit ≈ 255 cycles (162 ns) up to 24 MiB, DRAM ≈ 480–550 cycles (300–350 ns). Beyond the
L2 the repeats vary widely (64 MiB: 482–636; 1 GiB: 384–775 cycles): page / TLB effects and other users' memory
traffic on the shared device are possible causes, not measured. These replace `[ASSUMPTION]` cycle costs (docs/03 §9)
where T008 uses them.
