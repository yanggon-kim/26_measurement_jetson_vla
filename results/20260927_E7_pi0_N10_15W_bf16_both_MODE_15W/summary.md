# 20260927_E7_pi0_N10_15W_bf16_both_MODE_15W

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T01:55:30Z, power mode NV Power Mode: MODE_15W, model pi0, rung / mode bf16_both
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260927_E7_pi0_N10_15W --steps 10 --seconds 60.0 --modes eager,reduce-overhead --allow-capped`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | VDDQ_VDD2_1V8AO (regula… | 1765.9 | 403.8 |
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | VDD_CPU_CV | 0.0 | 1.7 |
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | VDD_GPU_SOC | 6044.4 | 2788.9 |
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | VIN_SYS_5V0 | 6048.4 | 3932.5 |
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | module | 13858.7 | 7126.9 |
| pi0 | bf16_both | compiled_subpath | MODE_15W | prefix | module_without_memory_r… | 12092.8 | 6723.1 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null compile_mode = set by the rung (run started before the writer fix); no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
