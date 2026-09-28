# 20260927_E7_smolvla_N10_15W_vlm_graphs_MODE_15W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T02:41:45Z, power mode NV Power Mode: MODE_15W, model smolvla, rung / mode vlm_graphs
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260927_E7_smolvla_N10_15W --steps 10 --seconds 60.0 --modes eager,reduce-overhead --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1512.9 | 403.9 |
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | VDD_CPU_CV | 1.9 | 2.0 |
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | VDD_GPU_SOC | 6156.8 | 2789.0 |
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | VIN_SYS_5V0 | 5646.5 | 3932.7 |
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | module | 13318.1 | 7127.6 |
| smolvla | vlm_graphs | compiled_subpath | MODE_15W | prefix | module_without_memory_r… | 11805.2 | 6723.7 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
