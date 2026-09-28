# 20260924_E7_groot_n16_eager_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T06:51:06Z, power mode NV Power Mode: MODE_50W, model groot_n16, rung / mode eager
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n16_eager_50W --version n16 --rung eager --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00T-N1.6-3B -`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | eager | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 2951.8 | 407.1 |
| groot_n16 | eager | eager | MODE_50W | prefix | VDD_CPU_CV | 812.5 | 407.0 |
| groot_n16 | eager | eager | MODE_50W | prefix | VDD_GPU_SOC | 17831.4 | 3734.4 |
| groot_n16 | eager | eager | MODE_50W | prefix | VIN_SYS_5V0 | 8106.6 | 4139.1 |
| groot_n16 | eager | eager | MODE_50W | prefix | module | 29702.4 | 8687.6 |
| groot_n16 | eager | eager | MODE_50W | prefix | module_without_memory_r… | 26750.6 | 8280.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
