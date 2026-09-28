# 20260924_E7_pi0_N10_bf16_both_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T09:11:28Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260924_E7_pi0_N10 --steps 10 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 2545.0 | 406.0 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1657.6 | 1224.9 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 24354.8 | 4785.2 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 7486.5 | 4175.4 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module | 36043.9 | 10591.5 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module_without_memory_r… | 33498.9 | 10185.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
