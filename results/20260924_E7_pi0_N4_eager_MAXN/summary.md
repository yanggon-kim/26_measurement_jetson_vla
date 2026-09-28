# 20260924_E7_pi0_N4_eager_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T07:46:03Z, power mode NV Power Mode: MAXN, model pi0, rung / mode none (eager)
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260924_E7_pi0_N4 --steps 4 --seconds 60.0 --modes eager`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | eager | eager | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3193.5 | 424.5 |
| pi0 | eager | eager | MAXN | prefix | VDD_CPU_CV | 2114.4 | 1392.2 |
| pi0 | eager | eager | MAXN | prefix | VDD_GPU_SOC | 37882.6 | 4780.9 |
| pi0 | eager | eager | MAXN | prefix | VIN_SYS_5V0 | 8533.3 | 4196.0 |
| pi0 | eager | eager | MAXN | prefix | module | 51723.8 | 10793.5 |
| pi0 | eager | eager | MAXN | prefix | module_without_memory_r… | 48530.3 | 10369.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
