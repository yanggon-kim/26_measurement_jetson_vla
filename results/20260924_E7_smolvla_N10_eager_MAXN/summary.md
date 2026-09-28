# 20260924_E7_smolvla_N10_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T08:24:50Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260924_E7_smolvla_N10 --steps 10 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 1906.0 | 458.6 |
| smolvla | eager | eager | MAXN | prefix | VDD_CPU_CV | 4039.9 | 1657.2 |
| smolvla | eager | eager | MAXN | prefix | VDD_GPU_SOC | 19285.2 | 4783.6 |
| smolvla | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 6575.3 | 4269.8 |
| smolvla | eager | eager | MAXN | prefix | module | 31806.3 | 11169.3 |
| smolvla | eager | eager | MAXN | prefix | module_without_memory_r… | 29900.4 | 10710.7 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
