# 20260924_E7_pi0_N4_bf16_both_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T09:02:40Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260924_E7_pi0_N4 --steps 4 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3139.9 | 445.2 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1919.0 | 1609.9 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 29962.8 | 4783.7 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 8433.4 | 4258.6 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module | 43455.2 | 11097.4 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module_without_memory_r… | 40315.2 | 10652.2 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
