# E0 over-current throttle, 60 s with temperatures (T002, after the review) — 2026-10-22, MAXN, jetson_clocks

bf16 4096³ GEMM loop for 60 s in 0.5 s slices (114 slices, 358 GEMMs each), after a 30 s rest.

| quantity | value |
|---|---|
| burst (first slice, no OC event) | 98.61 TFLOP/s at 57.5 °C (gpu-thermal) |
| sustained (median from the first OC event, slice 3, onward) | 93.06 TFLOP/s (−5.6 %) |
| last slice (60 s) | 91.93 TFLOP/s at 80.5 °C — still falling, no plateau within 60 s |
| VDD_GPU from slice 3 on | 98.9–99.7 W (the 5,060 mA ≈ 100 W current limit) |
| soctherm oc3 events | 16,529 total, rising from 112 to 184 per 0.5 s; equal to the counter difference between the gates before and after (no other load) |
| devfreq cur_freq | 1,575 MHz in every slice |
| correlation with temperature (slices 3–114) | TFLOP/s r = −0.959; oc3 events per slice r = +0.953 |

Reading (derived): at a fixed rail-current cap, the slowdown grows with die temperature (more power lost to leakage at
the same 100 W, more throttle events). No thermal limit is approached (80 °C), and devfreq never shows it.
Consequence: sustained heavy loads must be reported with their temperature and OC count; T003 reports burst and
sustained rates separately.
