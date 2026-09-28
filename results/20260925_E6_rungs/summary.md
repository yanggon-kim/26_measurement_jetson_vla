# 20260925_E6_rungs

* **What:** E6 — task T004
* **Conditions:** date 2026-09-25T21:26:17Z, model groot_n15
* **Command:** `python bench/roofline.py --outdir results/20260925_E6_rungs/groot_n15_all_compile_autotune`
* **Sub-runs:** groot_n15_all_compile_autotune, groot_n15_eager_tf32, groot_n16_all_compile_autotune, groot_n16_eager_tf32, groot_n17_ae_compile, groot_n17_eager_tf32, groot_n17_vlm_compile, pi05_bf16_both_graphs …

Headline — first rows of `groot_n15_all_compile_autotune/floor.csv` (6 rows):

| model | region | bytes_read | flops | peak_GBps | peak_TFLOPps | floor_mem_ms | floor_compute_ms |
|---|---|---|---|---|---|---|---|
| groot_n15 | vision | 855361408 | 0 | 204.8 | 42.0 | 4.1766 |  |
| groot_n15 | vision | 855361408 | 0 | 187.47 | 33.8506 | 4.5627 |  |
| groot_n15 | vlm_backbone | 3310714752 | 0 | 204.8 | 42.0 | 16.1656 |  |
| groot_n15 | vlm_backbone | 3310714752 | 0 | 187.47 | 33.8506 | 17.66 |  |
| groot_n15 | ae_step | 2137612288 | 0 | 204.8 | 42.0 | 10.4376 |  |
| groot_n15 | ae_step | 2137612288 | 0 | 187.47 | 33.8506 | 11.4024 |  |

**Anything odd:** notes: ["The roof is this Orin's own measured peaks (protocol §8); the spec percentages are given alongside and labelled, never used as the roof.", '[Orin] ncu rows take bytes from the L2 sysmem-miss sectors

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
