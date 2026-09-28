# 20260928_E7v_summary

* **What:** E7v (owner's request 2026-09-28): vision / language (VLM prefill) / action energy split, MAXN -- derived summary of the 12 E7v runs
* **Conditions:** date 2026-09-28T19:20:00Z, power mode MAXN (each input's meta.json)
* **Command:** `python3 bench/e7v_split.py results/20260928_E7v_summary`

Headline — first rows of `e7v_split.csv` (12 rows):

| run | model | config | vision_ms | language_ms | action_ms | full_ms | vision_mJ |
|---|---|---|---|---|---|---|---|
| 20260928_E7v_groot_n15_… | groot_n15 | all_compile_autotune | 31.8351 | 65.4681 | 60.8688 | 158.9956 | 1697.3 |
| 20260928_E7v_groot_n15_… | groot_n15 | eager_fp32 | 41.334 | 99.1817 | 110.5963 | 252.366 | 2079.0 |
| 20260928_E7v_groot_n16_… | groot_n16 | all_compile_autotune | 43.576 | 42.235 | 74.4898 | 163.1349 | 2291.6 |
| 20260928_E7v_groot_n16_… | groot_n16 | eager_fp32 | 61.7282 | 63.7856 | 215.822 | 336.049 | 2510.6 |
| 20260928_E7v_groot_n17_… | groot_n17 | eager_fp32 | 69.2708 | 85.712 | 213.762 | 364.2224 | 2585.2 |
| 20260928_E7v_groot_n17_… | groot_n17 | vlm_compile | 70.1307 | 85.7622 | 81.5509 | 241.7012 | 2586.0 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
