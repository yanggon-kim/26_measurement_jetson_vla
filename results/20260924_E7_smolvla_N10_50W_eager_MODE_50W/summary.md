# 20260924_E7_smolvla_N10_50W_eager_MODE_50W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-25T06:31:59Z, power mode NV Power Mode: MODE_50W, model smolvla, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260924_E7_smolvla_N10_50W --steps 10 --seconds 60.0 --modes eager --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | eager | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 1561.4 | 406.5 |
| smolvla | eager | eager | MODE_50W | prefix | VDD_CPU_CV | 1193.1 | 403.3 |
| smolvla | eager | eager | MODE_50W | prefix | VDD_GPU_SOC | 10942.8 | 3884.6 |
| smolvla | eager | eager | MODE_50W | prefix | VIN_SYS_5V0 | 5984.2 | 4139.2 |
| smolvla | eager | eager | MODE_50W | prefix | module | 19681.5 | 8833.6 |
| smolvla | eager | eager | MODE_50W | prefix | module_without_memory_r… | 18120.1 | 8427.1 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
