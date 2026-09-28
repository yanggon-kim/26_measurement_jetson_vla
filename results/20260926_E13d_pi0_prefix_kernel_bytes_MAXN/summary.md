# 20260926_E13d_pi0_prefix_kernel_bytes_MAXN

* **What:** T016 step 1d (EV-02c): per-kernel DRAM traffic of the pi0 bf16_both rung prefix + H1/H2 split
* **Conditions:** date 2026-09-27T03:16:42Z, power mode NV Power Mode: MAXN, rung / mode bf16_both (E7's, unmodified prefix incl. embed_image graph tree)

Headline — first rows of `kernel_bytes.csv` (2286 rows):

| frame | launch | kernel | grid | block | fills_B | writes_B | read_req_B |
|---|---|---|---|---|---|---|---|
| prefix_f0 | 0 | void at::<unnamed>::mul… | (3, 1, 1) | (512, 1, 1) | 606720 | 602528 | 605184 |
| prefix_f0 | 1 | triton_poi_fused__to_co… | (147, 1, 1) | (128, 1, 1) | 605312 | 301472 | 603136 |
| prefix_f0 | 2 | void implicit_convolve_… | (8, 36, 1) | (8, 8, 1) | 1684416 | 2359712 | 9966496 |
| prefix_f0 | 3 | triton_red_fused__to_co… | (36, 1, 1) | (128, 1, 1) | 1194592 | 28064 | 1198080 |
| prefix_f0 | 4 | triton_per_fused__to_co… | (8, 1, 1) | (128, 1, 1) | 35200 | 2464 | 28672 |
| prefix_f0 | 5 | triton_poi_fused__to_co… | (5, 16, 1) | (256, 1, 1) | 1201760 | 590240 | 1275136 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
