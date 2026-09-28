# 20260925_E6_pi0

* **What:** E6 — task T004
* **Conditions:** date 2026-09-25T15:03:48Z, model pi0
* **Command:** `python bench/roofline.py --outdir results/20260925_E6_pi0`

Headline — first rows of `roofline.csv` (303 rows):

| model | region_or_kernel | flops | bytes | ms | intensity | achieved_TFLOPps | achieved_GBps |
|---|---|---|---|---|---|---|---|
| pi0 | vision | 660529741824 | 1659214784 | 239.595051 | 398.0978 | 2.75686 | 6.925 |
| pi0 | vlm_prefix | 4094047420416 | 5017214976 | 232.930847 | 816.0 | 17.57624 | 21.54 |
| pi0 | ae_step | 38293352448 | 638046208 | 52.790832 | 60.0166 | 0.72538 | 12.086 |
| pi0 | void implicit_convolve_… |  | 8056992 |  |  |  |  |
| pi0 | void at::elementwise_ke… |  | 2375616 |  |  |  |  |
| pi0 | void at::vectorized_gat… |  | 2365440 |  |  |  |  |

**Anything odd:** notes: ["The roof is this Orin's own measured peaks (protocol §8); the spec percentages are given alongside and labelled, never used as the roof.", '[Orin] ncu rows take bytes from the L2 sysmem-miss sectors; raw_note: 2026-09-28 completeness audit: derived directory -- roofline.csv joins committed inputs (the E3/E4 latency runs and the C1 ncu CSVs named in 'inputs'); it has no raw files of its own, hence raw/ is em

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
