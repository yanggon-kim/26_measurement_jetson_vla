# 20260924_E7_smolvla_N10_vlm_graphs_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-24T09:16:08Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode vlm_graphs
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260924_E7_smolvla_N10 --steps 10 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3006.0 | 406.3 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1764.4 | 1227.2 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 36630.3 | 4783.7 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 8192.9 | 4168.4 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | module | 49593.6 | 10585.6 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | module_without_memory_r… | 46587.6 | 10179.3 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
