# E7-prep — do the four INA3221 rails double-count? (T005)

MAXN, GPU pinned 1300.5 MHz. Four sysfs rails sampled at 10 Hz for 60 s per load, after a 3 s
settle so no ramp is averaged in. mW = mV × mA from
`/sys/bus/i2c/drivers/ina3221/*/hwmon/*`; the "sum of shunt voltages" pseudo-channel is skipped.

| load | VDD_GPU_SOC | VDD_CPU_CV | VIN_SYS_5V0 | VDDQ_VDD2_1V8AO | sum |
|---|---|---|---|---|---|
| idle (absolute) | 4783 mW | 1210 mW | 4156 mW | 405 mW | 10555 mW |
| `clone_256mib` | +18681 | +1522 | +7265 | +4929 | +32396 mW |
| `copy_kernel_256mib` | +22174 | +1522 | +7849 | +5231 | +36776 mW |
| `gemm_8192_bf16` | +50720 | +1132 | +8802 | +5531 | +66186 mW |

**No double counting; E7 may sum the four.** The bf16 GEMM raises `VDD_GPU_SOC` by 50.7 W while
raising `VIN_SYS_5V0` by only 8.8 W — a rail cannot contain another and rise less than it.
`VDD_CPU_CV` *falls* from the copy loop (+1.52 W) to the GEMM loop (+1.13 W) while GPU power
doubles, so it is not a superset either, and `VDDQ_VDD2_1V8AO` tracks DRAM traffic (+4.93 W on
the pure-copy load) rather than compute.

**Odd / open:** the memory rail is not where memory traffic shows up most — the copy loop costs
+4.9 W there but +18.7 W on `VDD_GPU_SOC`, which also feeds the memory controller and fabric.
And the absolute scaling of `VDDQ_VDD2_1V8AO` is still unresolved: its INA3221 channel reports a
≈ 5 V bus voltage, not 1.8 V (question 2 in `STATUS.md`). The double-count conclusion does not
depend on that scaling.

A bandwidth-saturating load costs about half what a tensor-core load costs above idle
(32.4 W vs 66.2 W).
