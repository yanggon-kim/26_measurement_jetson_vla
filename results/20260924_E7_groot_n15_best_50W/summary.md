# 20260924_E7_groot_n15_best_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T07:05:07Z, power mode NV Power Mode: MODE_50W, model groot_n15, rung / mode best
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n15_best_50W --version n15 --rung best --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T-N1.5-3B --a`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | best | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 3666.4 | 405.0 |
| groot_n15 | best | eager | MODE_50W | prefix | VDD_CPU_CV | 802.7 | 408.0 |
| groot_n15 | best | eager | MODE_50W | prefix | VDD_GPU_SOC | 21793.7 | 3962.4 |
| groot_n15 | best | eager | MODE_50W | prefix | VIN_SYS_5V0 | 9210.3 | 4137.9 |
| groot_n15 | best | eager | MODE_50W | prefix | module | 35473.1 | 8913.2 |
| groot_n15 | best | eager | MODE_50W | prefix | module_without_memory_r… | 31806.7 | 8508.2 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
