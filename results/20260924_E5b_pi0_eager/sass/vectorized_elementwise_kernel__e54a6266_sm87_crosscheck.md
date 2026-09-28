# Static-vs-dynamic cross-check (sm_87) — `vectorized_elementwise_kernel__e54a6266`

## Static SASS counts (from `vectorized_elementwise_kernel__e54a6266_sm87_total_inventory.csv`)

- LSU        : **21** (12.5%)
- FMA        : **36** (21.4%)
- ALU        : **54** (32.1%)
- XU/Shuffle : **0** (0.0%)
- Other      : **57** (33.9%)
- TOTAL      : **168**

Static ratios:  LSU/FMA = 0.58x,  LSU/ALU = 0.39x,  ALU/FMA = 1.50x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260924_E5b_pi0_eager` | 0.42 | 0.21 | 1.49 | 0.00 | 0.42 | 0.00 | 2.00 | 0.29 |

**Direction agreement check.** Static LSU/FMA = 0.58x; dynamic ranges 2.00x-2.00x. Static counts predict the LSU-heavy character; dynamic ratio is 3.4x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
