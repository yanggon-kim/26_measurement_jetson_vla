# 20260926_E7_pi05_N10_30W_bf16_both_graphs_MODE_30W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-26T18:58:54Z, power mode NV Power Mode: MODE_30W, model pi05, rung / mode bf16_both_graphs
* **Command:** `python bench/power_phases.py --model pi05 --outdir-prefix results/20260926_E7_pi05_N10_30W --steps 10 --seconds 60.0 --modes eager,reduce-overhead --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | VDDQ_VDD2_1V8AO (regula… | 1585.0 | 409.8 |
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | VDD_CPU_CV | 641.8 | 411.4 |
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | VDD_GPU_SOC | 7718.1 | 3587.0 |
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | VIN_SYS_5V0 | 6096.4 | 4143.2 |
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | module | 16041.3 | 8551.4 |
| pi05 | bf16_both_graphs | compiled_subpath | MODE_30W | prefix | module_without_memory_r… | 14456.3 | 8141.5 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
