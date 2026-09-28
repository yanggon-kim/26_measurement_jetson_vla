## The answer

**GR00T N1.7's dominant action-head GEMM is memory-latency-bound, the first kernel of any
model in this campaign to meet §7's threshold.**

| mode | kernel | time in window | SM % | long-scoreboard, % of waiting | §7 verdict |
|---|---|---:|---:|---:|---|
| eager | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn` | 3304 µs | 37.7 | **61.6** | **memory-latency bound** |
| eager | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` | 3881 µs | 65.6 | 6.4 | compute bound (math-pipe throttle 14.9 % > 10 %) |
| best | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn` | 2908 µs | 41.5 | **61.0** | **memory-latency bound** |
| best | `Kernel2` (CUTLASS) | 2564 µs | 50.6 | 15.6 | neither; largest wait is `wait` 24.6 % |

For contrast, `reports/E5b_summary.md` found the six π0 / π0.5 / SmolVLA configurations at
**19.5–47.6 %** long scoreboard, time-weighted, with **none** reaching the 60 % bar. GR00T's
128×64 GEMM clears it in both modes. This agrees with the corrected E6: eager-to-eager,
GR00T's action step sits at 33 % of peak bandwidth against π0's 6.5 %, and compiled it
reaches 83–87 % (`reports/E6_groot_roofline.md`, correction block).
