# 20260928_E7v_smolvla_N10_vlm_graphs_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T18:15:23Z, power mode NV Power Mode: MAXN, model smolvla, rung / mode vlm_graphs
* **Command:** `python bench/power_phases.py --model smolvla --outdir-prefix results/20260928_E7v_smolvla_N10 --steps 10 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3311.3 | 414.7 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1601.9 | 1470.3 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 39519.3 | 4781.9 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 8726.0 | 4243.2 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | module | 53158.4 | 10910.1 |
| smolvla | vlm_graphs | compiled_subpath | MAXN | prefix | module_without_memory_r… | 49847.1 | 10495.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
