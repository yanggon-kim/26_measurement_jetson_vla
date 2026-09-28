# 20260925_E6_pi05

* **What:** E6 — task T004
* **Conditions:** date 2026-09-25T15:22:02Z, model pi05
* **Command:** `python bench/roofline.py --outdir results/20260925_E6_pi05`

Headline — first rows of `roofline.csv` (3 rows):

| model | region_or_kernel | flops | bytes | ms | intensity | achieved_TFLOPps | achieved_GBps |
|---|---|---|---|---|---|---|---|
| pi05 | vision | 660529741824 | 1659214784 | 103.6142 | 398.0978 | 6.3749 | 16.013 |
| pi05 | vlm_prefix | 4856517394432 | 10034126848 | 550.5063 | 484.0 | 8.82191 | 18.227 |
| pi05 | ae_step | 49983897600 | 1734819840 | 62.68983 | 28.8122 | 0.79732 | 27.673 |

**Anything odd:** notes: ["The roof is this Orin's own measured peaks (protocol §8); the spec percentages are given alongside and labelled, never used as the roof.", '[Orin] ncu rows take bytes from the L2 sysmem-miss sectors

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
