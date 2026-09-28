# 20260928_E7v_groot_n15_eager_fp32

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-28T19:02:24Z, power mode NV Power Mode: MAXN, model groot_n15, rung / mode eager_fp32
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260928_E7v_groot_n15_eager_fp32 --phases prefix,vision,vlm,ae,full --version n15 --rung eager_fp32 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkp`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 4684.1 | 471.4 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDD_CPU_CV | 2405.1 | 1596.7 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDD_GPU_SOC | 43245.9 | 4790.3 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VIN_SYS_5V0 | 10849.7 | 4323.4 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | module | 61184.8 | 11181.8 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | module_without_memory_r… | 56500.7 | 10710.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
