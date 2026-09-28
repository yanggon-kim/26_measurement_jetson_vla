# Static-vs-dynamic cross-check (sm_87) — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8`

## Static SASS counts (from `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87_total_inventory.csv`)

- LSU        : **634** (11.8%)
- FMA        : **1956** (36.5%)
- ALU        : **2001** (37.4%)
- XU/Shuffle : **1** (0.0%)
- Other      : **762** (14.2%)
- TOTAL      : **5354**

Static ratios:  LSU/FMA = 0.32x,  LSU/ALU = 0.32x,  ALU/FMA = 1.02x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260925_E5b_groot_n17_best` | 11.85 | 1.07 | 7.50 | 11.38 | 0.46 | 24.56 | 11.07 | 1.58 |

**Direction agreement check.** Static LSU/FMA = 0.32x; dynamic ranges 11.07x-11.07x. Static counts predict the LSU-heavy character; dynamic ratio is 34.2x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
