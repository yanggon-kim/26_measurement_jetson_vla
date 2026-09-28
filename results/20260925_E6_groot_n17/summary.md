# 20260925_E6_groot_n17

* **What:** E6 — task T004
* **Conditions:** date 2026-09-25T15:29:01Z, model groot_n17
* **Command:** `python bench/roofline.py --outdir results/20260925_E6_groot_n17`

Headline — first rows of `roofline.csv` (2 rows):

| model | region_or_kernel | flops | bytes | ms | intensity | achieved_TFLOPps | achieved_GBps |
|---|---|---|---|---|---|---|---|
| groot_n17 | vlm_backbone |  | 3047000064 | 161.0919 |  |  | 18.915 |
| groot_n17 | ae_step |  | 3241031936 | 20.0641 |  |  | 161.534 |

**Anything odd:** notes: ["The roof is this Orin's own measured peaks (protocol §8); the spec percentages are given alongside and labelled, never used as the roof.", '[Orin] ncu rows take bytes from the L2 sysmem-miss sectors

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
