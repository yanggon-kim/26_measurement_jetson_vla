# 20260927_E7_smolvla_N10_15W_eager_MODE_15W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T02:34:27Z, power mode NV Power Mode: MODE_15W, model smolvla, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260927_E7_smolvla_N10_15W --steps 10 --seconds 60.0 --modes eager --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | eager | eager | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1220.7 | 405.8 |
| smolvla | eager | eager | MODE_15W | prefix | VDD_CPU_CV | 395.4 | 2.0 |
| smolvla | eager | eager | MODE_15W | prefix | VDD_GPU_SOC | 5076.4 | 2788.8 |
| smolvla | eager | eager | MODE_15W | prefix | VIN_SYS_5V0 | 5112.0 | 3935.5 |
| smolvla | eager | eager | MODE_15W | prefix | module | 11804.6 | 7132.1 |
| smolvla | eager | eager | MODE_15W | prefix | module_without_memory_r… | 10583.9 | 6726.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
