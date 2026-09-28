# 20260928_E7v_groot_n17_eager_fp32

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-28T18:21:59Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_fp32
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260928_E7v_groot_n17_eager_fp32 --phases prefix,vision,vlm,ae,full --version n17 --rung eager_fp32 --checkpoint nvidia/GR00T-N1.7-3B --tf32 off`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n17 | eager_fp32 | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3509.3 | 459.5 |
| groot_n17 | eager_fp32 | eager | MAXN | prefix | VDD_CPU_CV | 2792.7 | 1574.4 |
| groot_n17 | eager_fp32 | eager | MAXN | prefix | VDD_GPU_SOC | 29290.9 | 4789.6 |
| groot_n17 | eager_fp32 | eager | MAXN | prefix | VIN_SYS_5V0 | 9007.3 | 4314.6 |
| groot_n17 | eager_fp32 | eager | MAXN | prefix | module | 44600.1 | 11138.1 |
| groot_n17 | eager_fp32 | eager | MAXN | prefix | module_without_memory_r… | 41090.8 | 10678.6 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
