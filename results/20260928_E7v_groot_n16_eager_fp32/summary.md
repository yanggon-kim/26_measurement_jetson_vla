# 20260928_E7v_groot_n16_eager_fp32

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-28T18:43:02Z, power mode NV Power Mode: MAXN, model groot_n16, rung / mode eager_fp32
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260928_E7v_groot_n16_eager_fp32 --phases prefix,vision,vlm,ae,full --version n16 --rung eager_fp32 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkp`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n16 | eager_fp32 | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3757.2 | 411.0 |
| groot_n16 | eager_fp32 | eager | MAXN | prefix | VDD_CPU_CV | 2796.1 | 1349.3 |
| groot_n16 | eager_fp32 | eager | MAXN | prefix | VDD_GPU_SOC | 34809.7 | 4782.3 |
| groot_n16 | eager_fp32 | eager | MAXN | prefix | VIN_SYS_5V0 | 9406.8 | 4238.7 |
| groot_n16 | eager_fp32 | eager | MAXN | prefix | module | 50769.8 | 10781.3 |
| groot_n16 | eager_fp32 | eager | MAXN | prefix | module_without_memory_r… | 47012.6 | 10370.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
