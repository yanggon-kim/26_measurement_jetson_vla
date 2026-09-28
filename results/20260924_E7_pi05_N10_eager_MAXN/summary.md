# 20260924_E7_pi05_N10_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T08:11:58Z, power mode NV Power Mode: MAXN, model pi05, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi05 --outdir-prefix results/20260924_E7_pi05_N10 --steps 10 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi05 | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 2503.7 | 428.1 |
| pi05 | eager | eager | MAXN | prefix | VDD_CPU_CV | 2004.6 | 1446.1 |
| pi05 | eager | eager | MAXN | prefix | VDD_GPU_SOC | 33520.1 | 4781.0 |
| pi05 | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 7563.1 | 4192.3 |
| pi05 | eager | eager | MAXN | prefix | module | 45591.5 | 10847.5 |
| pi05 | eager | eager | MAXN | prefix | module_without_memory_r… | 43087.8 | 10419.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
