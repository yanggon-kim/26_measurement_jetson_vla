# 20260926_E7_groot_n16_eager_30W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-26T19:23:55Z, power mode NV Power Mode: MODE_30W, model groot_n16, rung / mode eager
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260926_E7_groot_n16_eager_30W --version n16 --rung eager --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00T-N1.6-3B -`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | eager | eager | MODE_30W | prefix | VDDQ_VDD2_1V8AO (regula… | 2287.9 | 404.6 |
| groot_n16 | eager | eager | MODE_30W | prefix | VDD_CPU_CV | 804.0 | 404.7 |
| groot_n16 | eager | eager | MODE_30W | prefix | VDD_GPU_SOC | 11400.1 | 3587.0 |
| groot_n16 | eager | eager | MODE_30W | prefix | VIN_SYS_5V0 | 7235.0 | 4136.2 |
| groot_n16 | eager | eager | MODE_30W | prefix | module | 21727.1 | 8532.5 |
| groot_n16 | eager | eager | MODE_30W | prefix | module_without_memory_r… | 19439.2 | 8127.9 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
