# 20260928_E7v_smolvla_N10_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T18:07:46Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260928_E7v_smolvla_N10 --steps 10 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 1960.5 | 498.9 |
| smolvla | eager | eager | MAXN | prefix | VDD_CPU_CV | 2846.3 | 1603.3 |
| smolvla | eager | eager | MAXN | prefix | VDD_GPU_SOC | 19979.1 | 4857.0 |
| smolvla | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 6671.5 | 4336.3 |
| smolvla | eager | eager | MAXN | prefix | module | 31457.4 | 11295.4 |
| smolvla | eager | eager | MAXN | prefix | module_without_memory_r… | 29496.9 | 10796.5 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
