# 20260928_E7v_pi0_N10_bf16_both_MAXN

* **What:** E7 — task T004
* **Conditions:** date 2026-09-28T17:38:33Z, power mode NV Power Mode: MAXN, model pi0, rung / mode bf16_both
* **Command:** `python bench/power_phases.py --model pi0 --outdir-prefix results/20260928_E7v_pi0_N10 --steps 10 --seconds 60.0 --modes eager,reduce-overhead`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3921.7 | 444.5 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_CPU_CV | 1592.9 | 1465.5 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VDD_GPU_SOC | 37078.9 | 4800.1 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | VIN_SYS_5V0 | 9697.9 | 4292.4 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module | 52291.5 | 11002.5 |
| pi0 | bf16_both | compiled_subpath | MAXN | prefix | module_without_memory_r… | 48369.8 | 10558.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
