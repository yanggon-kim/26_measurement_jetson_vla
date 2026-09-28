# 20260927_E7_pi05_N10_15W_eager_MODE_15W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T02:14:21Z, power mode NV Power Mode: MODE_15W, model pi05, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi05 --outdir-prefix results/20260927_E7_pi05_N10_15W --steps 10 --seconds 60.0 --modes eager --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi05 | eager | eager | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1166.6 | 472.5 |
| pi05 | eager | eager | MODE_15W | prefix | VDD_CPU_CV | 228.0 | 62.7 |
| pi05 | eager | eager | MODE_15W | prefix | VDD_GPU_SOC | 5435.8 | 2789.0 |
| pi05 | eager | eager | MODE_15W | prefix | VIN_SYS_5V0 | 5063.7 | 4025.6 |
| pi05 | eager | eager | MODE_15W | prefix | module | 11894.0 | 7349.8 |
| pi05 | eager | eager | MODE_15W | prefix | module_without_memory_r… | 10727.4 | 6877.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
