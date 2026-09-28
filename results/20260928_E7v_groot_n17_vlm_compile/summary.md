# 20260928_E7v_groot_n17_vlm_compile

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-28T18:35:48Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260928_E7v_groot_n17_vlm_compile --phases prefix,vision,vlm,ae,full --version n17 --rung vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --tf32 on --head-mode r`

Headline — first rows of `power.csv` (30 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3342.4 | 444.2 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDD_CPU_CV | 2791.8 | 1550.1 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDD_GPU_SOC | 28330.9 | 4786.6 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VIN_SYS_5V0 | 8734.3 | 4297.3 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | module | 43199.5 | 11078.2 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | module_without_memory_r… | 39857.1 | 10634.0 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
