# 20260925_E12b_pi0_energy_N10_MAXN

* **What:** E12b M5 -- energy per frame, whole-frame loops (T015)
* **Conditions:** date 2026-09-25T19:31:30Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12b_m5.py --outdir results/20260925_E12b_pi0_energy_N10_MAXN --corun-config graphs_both --corun-split 12/4`

Headline — first rows of `power.csv` (18 rows):

| arm | variant | rail | mean_mW | idle_mW | above_idle_mW | period_ms | energy_per_frame_mJ |
|---|---|---|---|---|---|---|---|
| a_stock_full | bf16_both+stock+shippin… | VDD_GPU_SOC | 29240.14 | 4784.22 | 24455.92 | 424.6921 | 12418.058 |
| a_stock_full | bf16_both+stock+shippin… | VDD_CPU_CV | 1603.95 | 1394.14 | 209.81 | 424.6921 | 681.184 |
| a_stock_full | bf16_both+stock+shippin… | VIN_SYS_5V0 | 8417.94 | 4193.85 | 4224.08 | 424.6921 | 3575.0308 |
| a_stock_full | bf16_both+stock+shippin… | VDDQ_VDD2_1V8AO (regula… | 3112.73 | 428.18 | 2684.54 | 424.6921 | 1321.9506 |
| a_stock_full | bf16_both+stock+shippin… | module | 42374.75 | 10800.4 | 31574.35 | 424.6921 | 17996.2234 |
| a_stock_full | bf16_both+stock+shippin… | module_without_memory_r… | 39262.03 | 10372.22 | 28889.81 | 424.6921 | 16674.2728 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; layout_note: 2026-09-28 completeness audit: files moved from power.csv/power.csv and power.csv/raw/rail_samples.csv (a writer bug made power.csv a directory) to power.csv and raw/rail_samples.csv; contents unchang

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
