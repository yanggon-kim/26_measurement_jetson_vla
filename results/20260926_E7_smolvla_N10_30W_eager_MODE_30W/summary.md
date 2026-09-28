# 20260926_E7_smolvla_N10_30W_eager_MODE_30W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-26T19:03:36Z, power mode NV Power Mode: MODE_30W, model smolvla, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260926_E7_smolvla_N10_30W --steps 10 --seconds 60.0 --modes eager --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | eager | eager | MODE_30W | prefix | VDDQ_VDD2_1V8AO (regula… | 1472.4 | 407.0 |
| smolvla | eager | eager | MODE_30W | prefix | VDD_CPU_CV | 804.8 | 411.6 |
| smolvla | eager | eager | MODE_30W | prefix | VDD_GPU_SOC | 8505.1 | 3587.0 |
| smolvla | eager | eager | MODE_30W | prefix | VIN_SYS_5V0 | 5984.0 | 4140.4 |
| smolvla | eager | eager | MODE_30W | prefix | module | 16766.3 | 8546.0 |
| smolvla | eager | eager | MODE_30W | prefix | module_without_memory_r… | 15293.9 | 8139.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
