# 20260927_E7_groot_n17_eager_15W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-28T02:47:04Z, power mode NV Power Mode: MODE_15W, model groot_n17, rung / mode eager
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260927_E7_groot_n17_eager_15W --version n17 --rung eager --allow-capped --checkpoint nvidia/GR00T-N1.7-3B`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n17 | eager | eager | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1680.2 | 405.6 |
| groot_n17 | eager | eager | MODE_15W | prefix | VDD_CPU_CV | 1.3 | 3.0 |
| groot_n17 | eager | eager | MODE_15W | prefix | VDD_GPU_SOC | 5972.5 | 2781.4 |
| groot_n17 | eager | eager | MODE_15W | prefix | VIN_SYS_5V0 | 5820.6 | 3935.0 |
| groot_n17 | eager | eager | MODE_15W | prefix | module | 13474.5 | 7125.0 |
| groot_n17 | eager | eager | MODE_15W | prefix | module_without_memory_r… | 11794.4 | 6719.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
