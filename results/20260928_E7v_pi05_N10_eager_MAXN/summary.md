# 20260928_E7v_pi05_N10_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T17:49:59Z, power mode NV Power Mode: MAXN, model pi05, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi05 --outdir-prefix results/20260928_E7v_pi05_N10 --steps 10 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi05 | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 2653.8 | 481.9 |
| pi05 | eager | eager | MAXN | prefix | VDD_CPU_CV | 2032.5 | 1598.6 |
| pi05 | eager | eager | MAXN | prefix | VDD_GPU_SOC | 34149.4 | 4929.9 |
| pi05 | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 7842.3 | 4325.1 |
| pi05 | eager | eager | MAXN | prefix | module | 46677.9 | 11335.6 |
| pi05 | eager | eager | MAXN | prefix | module_without_memory_r… | 44024.1 | 10853.7 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
