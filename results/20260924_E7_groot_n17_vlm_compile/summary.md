# 20260924_E7_groot_n17_vlm_compile

* **What:** E7 (power and energy per phase) — task T009
* **Conditions:** date 2026-09-24T09:22:07Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --power --seconds 60 --outdir results/20260924_E7_groot_n17_vlm_compile --version n17 --rung vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --tf32 on --head-mode reduce-overhead --llm-mode max-autotu`

Headline — first rows of `power.csv` (18 rows):

| model | mode | exec_style | power_mode | phase | rail | mean_mW | idle_mW |
|---|---|---|---|---|---|---|---|
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDDQ_VDD2_1V8AO (regula… | 3225.9 | 406.1 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDD_CPU_CV | 2865.1 | 1219.6 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VDD_GPU_SOC | 27847.4 | 4784.3 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | VIN_SYS_5V0 | 8557.1 | 4154.5 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | module | 42495.5 | 10564.5 |
| groot_n17 | vlm_compile | compiled | MAXN | prefix | module_without_memory_r… | 39269.6 | 10158.4 |

**Anything odd:** notes: ["The memory rail is measured at the regulator INPUT (~5.04 V) and is NOT rescaled to 1.8 V. It is an upper bound on DRAM power; quote its above-idle delta next to the absolute value, and never call i; backfill_note: 2026-09-27: null gr00t_root = the default checkout; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
