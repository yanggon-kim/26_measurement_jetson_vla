# 20260927_E7_pi0_N10_15W_eager_MODE_15W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T01:39:12Z, power mode NV Power Mode: MODE_15W, model pi0, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260927_E7_pi0_N10_15W --steps 10 --seconds 60.0 --modes eager --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | eager | eager | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1392.3 | 409.6 |
| pi0 | eager | eager | MODE_15W | prefix | VDD_CPU_CV | 148.7 | 7.0 |
| pi0 | eager | eager | MODE_15W | prefix | VDD_GPU_SOC | 5780.5 | 2789.0 |
| pi0 | eager | eager | MODE_15W | prefix | VIN_SYS_5V0 | 5400.5 | 3942.0 |
| pi0 | eager | eager | MODE_15W | prefix | module | 12721.9 | 7147.6 |
| pi0 | eager | eager | MODE_15W | prefix | module_without_memory_r… | 11329.6 | 6738.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
