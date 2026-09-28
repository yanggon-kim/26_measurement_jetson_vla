# 20260928_E7v_pi0_N10_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T17:27:38Z, power mode NV Power Mode: MAXN, model pi0, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260928_E7v_pi0_N10 --steps 10 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3315.4 | 413.2 |
| pi0 | eager | eager | MAXN | prefix | VDD_CPU_CV | 2005.6 | 1273.9 |
| pi0 | eager | eager | MAXN | prefix | VDD_GPU_SOC | 38581.9 | 4782.2 |
| pi0 | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 8759.5 | 4226.3 |
| pi0 | eager | eager | MAXN | prefix | module | 52662.4 | 10695.5 |
| pi0 | eager | eager | MAXN | prefix | module_without_memory_r… | 49346.9 | 10282.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
