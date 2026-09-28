# Static-vs-dynamic cross-check (sm_87) — `CatArrayBatchedCopy__73a6cb7a`

## Static SASS counts (from `CatArrayBatchedCopy__73a6cb7a_sm87_total_inventory.csv`)

- LSU        : **13** (4.5%)
- FMA        : **103** (35.8%)
- ALU        : **121** (42.0%)
- XU/Shuffle : **9** (3.1%)
- Other      : **42** (14.6%)
- TOTAL      : **288**

Static ratios:  LSU/FMA = 0.13x,  LSU/ALU = 0.11x,  ALU/FMA = 1.17x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260924_E5b_pi05_eager` | 2.77 | 13.37 | 27.80 | 1.39 | 1.39 | 1.00 | 0.21 | 0.10 |

**Direction agreement check.** Static LSU/FMA = 0.13x; dynamic ranges 0.21x-0.21x. Static counts predict the LSU-heavy character; dynamic ratio is 1.6x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
