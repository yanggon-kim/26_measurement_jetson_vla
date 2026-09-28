# Static-vs-dynamic cross-check (sm_87) — `triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5`

## Static SASS counts (from `triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5_sm87_total_inventory.csv`)

- LSU        : **5** (5.2%)
- FMA        : **23** (24.0%)
- ALU        : **23** (24.0%)
- XU/Shuffle : **0** (0.0%)
- Other      : **45** (46.9%)
- TOTAL      : **96**

Static ratios:  LSU/FMA = 0.22x,  LSU/ALU = 0.22x,  ALU/FMA = 1.00x

## Dynamic ncu pipe utilisation (single-config — one ncu run)

> Only one configuration is profiled. The table below shows the single run's dynamic counters. For workloads where multiple inputs or batch sizes are profiled separately, re-run this with a top-level `{run_id: {kernel_name: ...}}` JSON to compare across runs.

| run | LSU% | FMA% | ALU% | LD% | WR% | LD:WR | dyn LSU/FMA | dyn LSU/ALU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260924_E5b_pi05_best` | 9.42 | 8.24 | 19.62 | 5.37 | 4.05 | 1.33 | 1.14 | 0.48 |

**Direction agreement check.** Static LSU/FMA = 0.22x; dynamic ranges 1.14x-1.14x. Static counts predict the LSU-heavy character; dynamic ratio is 5.3x amplified — see why below.

## Why static ≠ dynamic

- **Static counts include every SASS line emitted by the compiler**, including code paths that fire only on boundary cases.
- **Predication.** Many ALU/branch ops gate body code that only fires on boundary rows or sentinel iterations. The static count counts them once; the dynamic counter counts them per warp-execution.
- **Looped vs unrolled execution.** Long-span backward BRAs are outer iteration loops that execute many times; short-span ones are bisection / spin / barrier waits that fire on contention only. Fully-unrolled inner blocks execute once per warp tile. Static count reflects source-text layout, not weighted-by-trip-count execution.
- **Per-pipe peak issue rate.** `pct_of_peak_sustained_active` is normalised by the maximum issuable per cycle for that pipe, which differs across pipes (LSU has its own issue slot; ALU and FMA share some sub-pipes). 100% LSU is a different denominator than 100% FMA, so dividing dynamic LSU% / FMA% is not the same as dividing static LSU count / FMA count.
- **Per-instruction issue cost.** Vector LDGs count as one *instruction* but burn multiple LSU cycles; NOPs and uniform-datapath ops consume zero pipe slots. Static and dynamic are measuring different things.

**Bottom line.** Static counts give the *direction* of pipe utilisation (LSU-heavy / FMA-heavy / ALU-heavy); only the dynamic counter gives the magnitude. If the static and dynamic directions disagree (e.g. static says ALU-heavy but dynamic says LSU-heavy), check whether the ALU instructions are inside a predicated boundary path that rarely fires.
