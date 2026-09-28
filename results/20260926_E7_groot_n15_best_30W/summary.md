# 20260926_E7_groot_n15_best_30W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-26T19:37:53Z, power mode NV Power Mode: MODE_30W, model groot_n15, rung / mode best
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260926_E7_groot_n15_best_30W --version n15 --rung best --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T-N1.5-3B --a`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | best | eager | MODE_30W | prefix | VDDQ_VDD2_1V8AO (regula… | 2637.3 | 405.2 |
| groot_n15 | best | eager | MODE_30W | prefix | VDD_CPU_CV | 417.6 | 407.0 |
| groot_n15 | best | eager | MODE_30W | prefix | VDD_GPU_SOC | 12056.8 | 3589.7 |
| groot_n15 | best | eager | MODE_30W | prefix | VIN_SYS_5V0 | 7839.5 | 4137.8 |
| groot_n15 | best | eager | MODE_30W | prefix | module | 22951.3 | 8539.6 |
| groot_n15 | best | eager | MODE_30W | prefix | module_without_memory_r… | 20314.0 | 8134.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
