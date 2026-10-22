# A hidden memory-path performance state after sustained load (T003 finding) — 2026-10-22, MAXN, jetson_clocks, gate passed

**What:** a cold weight stream (`F.linear(x[51, 1024], W_i)` over 18 distinct 8 MiB weights, 144 MiB) takes **≈ 50 µs
per call (≈ 166 GB/s)** normally, and **≈ 34 µs (≈ 245 GB/s)** in a second state. The second state appears after
tens of seconds of sustained heavy GPU load and ends a few seconds after the load stops.

| run (`state_probe_load90.csv`, one probe per 2 s) | probe µs / call | weight GB/s | VDD_GPU | GPU °C | oc3 / 2 s |
|---|---|---|---|---|---|
| idle (0–20 s) | 50.0–52.7 | 159–168 | 7.1 W | 38 | 0 |
| load, 0–50 s in (bf16 4096³ back to back) | 49.7–50.9 | 165–169 | 90–93 W | 57 → 73 | 0 |
| load, from ≈ 55 s in | **34.2–34.9** | **240–245** | 97 → 99.3 W | 76 → 79 | 0, then 399–446 |
| recovery, first ≈ 2 s | 34.4 | 244 | 8.7 W | 61 | 0 |
| recovery, from ≈ 10 s | 50.0–52.7 | 159–168 | 7.1 W | ≤ 55 | 0 |

- The shorter run (`state_probe.csv`: 30 s of load at 91–92 W) never left the normal state.
- The over-current events start *after* the switch: they are a consequence of the higher power drawn in the fast state,
  not its cause.
- **No clock changes.** Every clock in `/sys/kernel/debug/clk/clk_summary` (`raw/clk_idle|during|after.txt`), the BPMP
  `emc` (4266 MHz) and `mcf` (1503 MHz) rates, and devfreq (1575 MHz) are identical in both states. The mechanism is
  not identified: it is invisible to every clock and counter I can read.
- **Effect on the π0 expert** (`raw/e2_after_load/`, E2 run right after a 60 s load): gate / up 33.9 µs vs 49.6 normal,
  down 33.2 vs 36.4, q 18.0 vs 20.1. That is a 1.26× faster projection stack per denoising step (2.61 vs 3.29 ms).
- **int8 E1 GEMM:** the same kind of switch, 258 → 313–331 TOP/s at the same clock
  (`results/20261022_E1_peak_backtoback/`).
- **Consequence for every later timing run:** `bench/gate.py: memory_state_probe()` classifies the state (normal
  ≈ 50 µs, fast < 42 µs, ≈ 3 ms of GPU work) before and after each measured block. Runs report which state they ran
  in; a block that changes state mid-way is flagged.
