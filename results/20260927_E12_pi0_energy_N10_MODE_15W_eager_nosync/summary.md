# 20260927_E12_pi0_energy_N10_MODE_15W_eager_nosync

* **What:** E12 Step 5 -- energy of the pipelined vs the serial schedule
* **Conditions:** power mode NV Power Mode: MODE_15W, model pi0, rung / mode eager
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun_energy.py --allow-capped --steps 10 --frames 10 --seconds 60 --outdir results/20260927_E12_pi0_energy_N10_MODE_15W_eager_nosync`

Headline — first rows of `power.csv` (12 rows):

| model | mode | mechanism | phase | rail | power_mode | sm_count | N |
|---|---|---|---|---|---|---|---|
| pi0 | eager | pi0_nosync | pipelined | VDDQ_VDD2_1V8AO (regula… | NV Power Mode: MODE_15W | 6 | 10 |
| pi0 | eager | pi0_nosync | pipelined | VDD_CPU_CV | NV Power Mode: MODE_15W | 6 | 10 |
| pi0 | eager | pi0_nosync | pipelined | VDD_GPU_SOC | NV Power Mode: MODE_15W | 6 | 10 |
| pi0 | eager | pi0_nosync | pipelined | VIN_SYS_5V0 | NV Power Mode: MODE_15W | 6 | 10 |
| pi0 | eager | pi0_nosync | pipelined | module | NV Power Mode: MODE_15W | 6 | 10 |
| pi0 | eager | pi0_nosync | pipelined | module_without_memory_r… | NV Power Mode: MODE_15W | 6 | 10 |

**Anything odd:** nosync_note: pi0_nosync: memoised the constant att_masks tensors at modeling_pi0.py:681 (per prefix) and :746 (per denoise step); 1+N host syncs per frame removed; not_measured: ["Energy of a pipeline that ACTUALLY overlaps -- none exists on this device, see the report's §1 and §5. This arm measures the schedule as it really runs.", 'Per-stage energy attribution within a fram; layout_note: 2026-09-28 completeness audit: files moved from power.csv/power.csv and power.csv/raw/rail_samples.csv (a writer bug made power.csv a directory) to power.csv and raw/rail_samples.csv; contents unchang

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
