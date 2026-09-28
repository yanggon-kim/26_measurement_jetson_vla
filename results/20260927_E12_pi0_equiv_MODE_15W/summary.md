# 20260927_E12_pi0_equiv_MODE_15W

* **What:** E12 fresh-KV equivalence (spec §6)
* **Conditions:** date 2026-09-28T04:58:44Z, power mode NV Power Mode: MODE_15W
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun.py --allow-capped --mode equivalence --path eager --nosync --steps 10 --frames 6 --outdir results/20260927_E12_pi0_equiv_MODE_15W`

Headline — first rows of `equivalence.csv` (11 rows):

| mechanism | path | frame | obs_id | seed | max_abs_diff | max_rel_diff | bitwise_identical |
|---|---|---|---|---|---|---|---|
| pipeline | eager | 0 | obs_0 | 1234 | 0.0 | 0.0 | True |
| pipeline | eager | 1 | obs_1 | 1235 | 0.0 | 0.0 | True |
| stale_control | eager | 1 | obs_1 | 1235 | 0.12436322122812271 | 20.530189514160156 | False |
| pipeline | eager | 2 | obs_2 | 1236 | 0.0 | 0.0 | True |
| stale_control | eager | 2 | obs_2 | 1236 | 0.0831499695777893 | 20.155271530151367 | False |
| pipeline | eager | 3 | obs_3 | 1237 | 0.0 | 0.0 | True |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
