# 20260924_E7_groot_n16_all_compile_autotune

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-24T09:28:28Z, power mode NV Power Mode: MAXN, model groot_n16, rung / mode all_compile_autotune
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n16_all_compile_autotune --version n16 --rung all_compile_autotune --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3823.5 | 404.9 |
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | VDD_CPU_CV | 2588.7 | 1219.5 |
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | VDD_GPU_SOC | 43586.6 | 4782.6 |
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | VIN_SYS_5V0 | 9502.3 | 4161.6 |
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | module | 59501.0 | 10568.6 |
| groot_n16 | all_compile_autotune | compiled | MAXN | prefix | module_without_memory_r… | 55677.5 | 10163.7 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
