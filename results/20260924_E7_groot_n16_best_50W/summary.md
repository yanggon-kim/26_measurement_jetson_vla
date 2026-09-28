# 20260924_E7_groot_n16_best_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T06:55:45Z, power mode NV Power Mode: MODE_50W, model groot_n16, rung / mode best
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n16_best_50W --version n16 --rung best --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00T-N1.6-3B --e`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | best | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 2945.1 | 404.3 |
| groot_n16 | best | eager | MODE_50W | prefix | VDD_CPU_CV | 804.5 | 405.3 |
| groot_n16 | best | eager | MODE_50W | prefix | VDD_GPU_SOC | 17780.4 | 3799.5 |
| groot_n16 | best | eager | MODE_50W | prefix | VIN_SYS_5V0 | 8086.6 | 4136.5 |
| groot_n16 | best | eager | MODE_50W | prefix | module | 29616.6 | 8745.7 |
| groot_n16 | best | eager | MODE_50W | prefix | module_without_memory_r… | 26671.5 | 8341.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
