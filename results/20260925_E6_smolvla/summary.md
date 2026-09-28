# 20260925_E6_smolvla

* **What:** E6 — task T004
* **Conditions:** date 2026-09-25T15:22:12Z, model smolvla
* **Command:** `python bench/roofline.py --outdir results/20260925_E6_smolvla`

Headline — first rows of `roofline.csv` (3 rows):

| model | region_or_kernel | flops | bytes | ms | intensity | achieved_TFLOPps | achieved_GBps |
|---|---|---|---|---|---|---|---|
| smolvla | vision |  | 196459008 | 33.257 |  |  | 5.907 |
| smolvla | vlm_prefix | 72437915520 | 409253760 | 64.4909 | 177.0 | 1.12323 | 6.346 |
| smolvla | ae_step | 10685696000 | 203392000 | 60.80137 | 52.5374 | 0.17575 | 3.345 |

**Anything odd:** notes: ["The roof is this Orin's own measured peaks (protocol §8); the spec percentages are given alongside and labelled, never used as the roof.", '[Orin] ncu rows take bytes from the L2 sysmem-miss sectors

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
