# 20260924_E7_groot_n15_eager_fp32_50W

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-25T07:43:16Z, power mode NV Power Mode: MODE_50W, model groot_n15, rung / mode eager_fp32
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n15_eager_fp32_50W --version n15 --rung eager_fp32 --allow-capped --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 3702.4 | 407.0 |
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | VDD_CPU_CV | 802.1 | 409.3 |
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | VDD_GPU_SOC | 21848.3 | 3942.8 |
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | VIN_SYS_5V0 | 9249.2 | 4141.7 |
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | module | 35602.0 | 8900.8 |
| groot_n15 | eager_fp32 | eager | MODE_50W | prefix | module_without_memory_r… | 31899.6 | 8493.8 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
