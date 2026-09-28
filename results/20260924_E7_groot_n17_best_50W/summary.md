# 20260924_E7_groot_n17_best_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T06:46:23Z, power mode NV Power Mode: MODE_50W, model groot_n17, rung / mode best
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n17_best_50W --version n17 --rung best --allow-capped --checkpoint nvidia/GR00T-N1.7-3B`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n17 | best | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 2725.5 | 405.6 |
| groot_n17 | best | eager | MODE_50W | prefix | VDD_CPU_CV | 1054.2 | 405.7 |
| groot_n17 | best | eager | MODE_50W | prefix | VDD_GPU_SOC | 15370.5 | 3709.3 |
| groot_n17 | best | eager | MODE_50W | prefix | VIN_SYS_5V0 | 7715.6 | 4137.6 |
| groot_n17 | best | eager | MODE_50W | prefix | module | 26865.8 | 8658.2 |
| groot_n17 | best | eager | MODE_50W | prefix | module_without_memory_r… | 24140.2 | 8252.6 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null gr00t_root = the default checkout; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
