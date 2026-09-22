# E7-prep — do the four INA3221 rails double-count? (T005)

MAXN, GPU pinned 1300.5 MHz. Four sysfs rails sampled at 10 Hz for 60 s per load, after a 3 s
settle so no ramp is averaged in. mW = mV × mA from
`/sys/bus/i2c/drivers/ina3221/*/hwmon/*`; the "sum of shunt voltages" pseudo-channel is skipped.

| load | VDD_GPU_SOC | VDD_CPU_CV | VIN_SYS_5V0 | VDDQ_VDD2_1V8AO | sum |
|---|---|---|---|---|---|
| idle (absolute) | 4783 mW | 1195 mW | 4115 mW | 385 mW | 10478 mW |
| `clone_256mib` | +18704 | +1503 | +7361 | +4953 | +32522 mW |
| `copy_kernel_256mib` | +22191 | +1512 | +7855 | +5228 | +36786 mW |
| `gemm_8192_bf16` | +51784 | +1160 | +8674 | +5422 | +67040 mW |

**No double counting; E7 may sum the four.** One comparison carries it: the bf16 GEMM raises
`VDD_GPU_SOC` by 51.8 W while raising `VIN_SYS_5V0` by only 8.7 W — a rail cannot contain
another and rise less than it. `VDDQ_VDD2_1V8AO` tracks DRAM traffic (+4.95 W on the pure-copy
load) rather than compute.

**Do not** use the apparent fall in `VDD_CPU_CV` from the copy loop to the GEMM loop as a second
argument: those are means, and the medians above idle are 1593 / 1593 / 1589 mW, i.e.
indistinguishable. The CPU rail simply does not respond to GPU load.

**Open:** the absolute scaling of `VDDQ_VDD2_1V8AO` — its INA3221 channel reports a ≈ 5 V bus
voltage, not 1.8 V (question 2 in `STATUS.md`). The double-count conclusion does not depend on
it. A bandwidth-saturating load costs about half what a tensor-core load costs above idle
(32.5 W vs 67.0 W).
