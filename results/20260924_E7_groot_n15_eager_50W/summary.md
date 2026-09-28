# 20260924_E7_groot_n15_eager_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T07:00:27Z, power mode NV Power Mode: MODE_50W, model groot_n15, rung / mode eager
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n15_eager_50W --version n15 --rung eager --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T-N1.5-3B -`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | eager | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 3656.4 | 407.4 |
| groot_n15 | eager | eager | MODE_50W | prefix | VDD_CPU_CV | 803.6 | 411.5 |
| groot_n15 | eager | eager | MODE_50W | prefix | VDD_GPU_SOC | 21790.4 | 3917.8 |
| groot_n15 | eager | eager | MODE_50W | prefix | VIN_SYS_5V0 | 9194.0 | 4140.1 |
| groot_n15 | eager | eager | MODE_50W | prefix | module | 35444.4 | 8876.7 |
| groot_n15 | eager | eager | MODE_50W | prefix | module_without_memory_r… | 31788.0 | 8469.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
