# 20260924_E7_pi0_N10_50W_bf16_both_MODE_50W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-25T06:06:33Z, power mode NV Power Mode: MODE_50W, model pi0, rung / mode bf16_both
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260924_E7_pi0_N10_50W --steps 10 --seconds 60.0 --modes eager,reduce-overhead --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | VDDQ_VDD2_1V8AO (regula… | 2673.0 | 405.4 |
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | VDD_CPU_CV | 557.0 | 403.3 |
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | VDD_GPU_SOC | 16146.7 | 3924.6 |
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | VIN_SYS_5V0 | 7634.8 | 4136.8 |
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | module | 27011.5 | 8870.2 |
| pi0 | bf16_both | compiled_subpath | MODE_50W | prefix | module_without_memory_r… | 24338.5 | 8464.7 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null rung / compile_mode replaced by their meaning (eager = none; rung runs = set by the rung); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
