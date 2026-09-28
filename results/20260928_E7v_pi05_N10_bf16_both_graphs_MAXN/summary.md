# 20260928_E7v_pi05_N10_bf16_both_graphs_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T18:01:09Z, power mode NV Power Mode: MAXN, model pi05, rung / mode bf16_both_graphs
* **Command:** `python bench/power_phases.py --model pi05 --outdir-prefix results/20260928_E7v_pi05_N10 --steps 10 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3790.5 | 508.8 |
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1598.2 | 1608.3 |
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 37938.4 | 4872.4 |
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 9517.9 | 4357.1 |
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | module | 52845.0 | 11346.6 |
| pi05 | bf16_both_graphs | compiled_subpath | MAXN | prefix | module_without_memory_r… | 49054.5 | 10837.8 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
