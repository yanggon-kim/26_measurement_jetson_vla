# 20260926_E7_groot_n16_best_30W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-26T19:28:33Z, power mode NV Power Mode: MODE_30W, model groot_n16, rung / mode best
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260926_E7_groot_n16_best_30W --version n16 --rung best --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00T-N1.6-3B --e`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | best | eager | MODE_30W | prefix | VDDQ_VDD2_1V8AO (regula… | 2292.2 | 404.6 |
| groot_n16 | best | eager | MODE_30W | prefix | VDD_CPU_CV | 826.1 | 404.7 |
| groot_n16 | best | eager | MODE_30W | prefix | VDD_GPU_SOC | 11416.8 | 3586.8 |
| groot_n16 | best | eager | MODE_30W | prefix | VIN_SYS_5V0 | 7241.2 | 4136.4 |
| groot_n16 | best | eager | MODE_30W | prefix | module | 21776.3 | 8532.6 |
| groot_n16 | best | eager | MODE_30W | prefix | module_without_memory_r… | 19484.1 | 8128.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
