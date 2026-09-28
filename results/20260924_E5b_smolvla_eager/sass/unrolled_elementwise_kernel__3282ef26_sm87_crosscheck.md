# Static-vs-dynamic cross-check (sm_87) — `unrolled_elementwise_kernel__3282ef26`

## Static SASS counts (from `unrolled_elementwise_kernel__3282ef26_sm87_total_inventory.csv`)

- LSU        : **178** (9.1%)
- FMA        : **256** (13.1%)
- ALU        : **908** (46.5%)
- XU/Shuffle : **0** (0.0%)
- Other      : **610** (31.2%)
- TOTAL      : **1952**

Static ratios:  LSU/FMA = 0.70x,  LSU/ALU = 0.20x,  ALU/FMA = 3.55x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260924_E5b_smolvla_eager` | 0.43 | 0.25 | 1.23 | 0.22 | 0.22 | 1.00 | 1.71 | 0.35 |

**Direction agreement check.** Static LSU/FMA = 0.70x; dynamic ranges 1.71x-1.71x. Static counts predict the LSU-heavy character; dynamic ratio is 2.5x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
