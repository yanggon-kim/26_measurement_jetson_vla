# Static-vs-dynamic cross-check (sm_87) — `elementwise_kernel__76a18c20`

## Static SASS counts (from `elementwise_kernel__76a18c20_sm87_total_inventory.csv`)

- LSU        : **6** (0.7%)
- FMA        : **333** (38.5%)
- ALU        : **291** (33.7%)
- XU/Shuffle : **108** (12.5%)
- Other      : **126** (14.6%)
- TOTAL      : **864**

Static ratios:  LSU/FMA = 0.02x,  LSU/ALU = 0.02x,  ALU/FMA = 0.87x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260924_E5b_smolvla_eager` | 0.15 | 0.19 | 1.41 | 0.10 | 0.05 | 2.00 | 0.83 | 0.11 |

**Direction agreement check.** Static LSU/FMA = 0.02x; dynamic ranges 0.83x-0.83x. Static counts predict the LSU-heavy character; dynamic ratio is 46.1x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
