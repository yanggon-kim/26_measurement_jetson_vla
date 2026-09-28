# 20260924_E7_groot_n15_eager_fp32

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-24T08:46:52Z, power mode NV Power Mode: MAXN, model groot_n15, rung / mode eager_fp32
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n15_eager_fp32 --version n15 --rung eager_fp32 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T-N1.5-3B --api old `

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 4491.2 | 496.0 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDD_CPU_CV | 2536.3 | 1763.7 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VDD_GPU_SOC | 42461.2 | 4796.2 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | VIN_SYS_5V0 | 10498.5 | 4306.3 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | module | 59987.2 | 11362.2 |
| groot_n15 | eager_fp32 | eager | MAXN | prefix | module_without_memory_r… | 55496.1 | 10866.2 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
