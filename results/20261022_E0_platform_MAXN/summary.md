# E0 platform record (T002) — 2026-10-22, MAXN (id 0), jetson_clocks applied

| item | value | file |
|---|---|---|
| gate (protocol §1) | all 5 checks pass before every step | `raw/gate_before_*.json` |
| GPU | Thor, CC 11.0, `sm_110`, 20 SMs, L2 33,554,432 B, smem/SM 233,472 B, 48 warps/SM, 65,536 regs/SM | `gpu_properties.json` |
| locked GPU clock under load | **1,575,000,000 Hz**, constant: 200/200 devfreq samples during a 10 s bf16 GEMM loop | `clock_probe.csv` |
| EMC | 4,266,000,000 Hz | `raw/cmd_emc_rate.txt` |
| CPU | 14 online, all at 2.601 GHz (min = max), schedutil | `raw/cmd_cpu_governors_freqs.txt` |
| memory | 125,772 MB; swap 0; no zram | `raw/cmd_free.txt`, `raw/cmd_swapon.txt` |
| idle power, 60 s at 10 Hz (mean W) | VIN 33.27 (module power [ASSUMPTION]) · VDD_GPU 7.10 · VDD_CPU_SOC_MSS 8.72 · VIN_SYS_5V0 5.85 | `idle_power_summary.csv` |
| GPU power limit | VDD_GPU warning limit 5,060 mA ≈ 100 W; soctherm oc3 throttles above it | `raw/power_limits.txt` |
| effect of the limit | bf16 4096³ GEMM loop: 99.5 TFLOP/s for the first 0.5 s, then **94.0 TFLOP/s** sustained (−5.5 %), 5,245 oc3 events in 20 s; devfreq still 1575 MHz | `oc_throttle_probe_bf16.csv` |

Odd:
- devfreq cannot see the over-current throttle. The clock probe counts as throttled (1,566 oc3 events), although every
  devfreq sample read 1575 MHz.
- The oc counters were already nonzero before T002 (oc2 527, oc3 ≈ 49,000), probably from other users' loads.
- ina3221 current steps are 20 mA (≈ 0.40 W per sample on VDD_GPU).
