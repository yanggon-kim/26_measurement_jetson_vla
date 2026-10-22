# E1 back-to-back ceilings (T003, review follow-up) — 2026-10-22, MAXN, gate passed before each probe, 30 s rest between

A @ B with distinct operands at each dtype's best E1 shape, issued back to back for 30 s in 0.5 s slices
(`bench/oc_throttle_probe.py --operands distinct`). E1's isolated-call rates synchronised after every call.

| dtype, shape | first slice | fastest slice | median of all slices | median from the first OC event | oc3 events | GPU °C at the end |
|---|---|---|---|---|---|---|
| bf16 2048³ | 111.3 | 111.3 (slice 0) | 104.3 | 104.308 | 14806 | 73.69 |
| fp16 2048³ | 109.1 | 109.1 (slice 0) | 95.2 | 95.196 | 7853 | 76.66 |
| tf32 1024³ | 61.8 | 61.9 (slice 57) | 61.8 | no OC event | 0 | 75.5 |
| fp32 16384³ | 6.4 | 6.4 (slice 1) | 6.4 | no OC event | 0 | 65.69 |
| int8 8192³ | 258.0 | 330.6 (slice 41) | 258.1 | 313.139 | 1550 | 77.41 |
| fp8_e4m3 4096³ | 197.0 | 197.0 (slice 0) | 192.7 | 192.6565 | 9406 | 77.59 |
| bf16 4096³ | 85.7 | 94.2 (slice 47) | 85.7 | 91.303 | 1430 | 77.91 |

Tensor rates are state-dependent. int8 ran at 258 TOP/s for 20 s, then at 313–331 TOP/s, with VDD_GPU 83 → 99.7 W at
the same 1575 MHz devfreq. bf16 / fp16 / fp8 ran slower back to back than E1's isolated calls (129.8 / 129.8 / 302.9).
`hw_constants.json` takes the highest rate measured in either mode and lists both, plus the sustained rate.
See `results/20261022_E1_state_probe/` for the memory-path state switch.
