# 20260926_E12_pi0_equiv_MODE_30W

* **What:** E12 fresh-KV equivalence (spec §6) at MODE_30W
* **Conditions:** date 2026-09-26T21:18:06Z, power mode NV Power Mode: MODE_30W
* **Command:** `bench/corun.py --allow-capped --mode equivalence --path eager --nosync --steps 10 --frames 6 --outdir results/20260926_E12_pi0_equiv_MODE_30W`

Headline — first rows of `equivalence.csv` (11 rows):

| mechanism | path | frame | obs_id | seed | max_abs_diff | max_rel_diff | bitwise_identical |
|---|---|---|---|---|---|---|---|
| pipeline | eager | 0 | obs_0 | 1234 | 0.0 | 0.0 | True |
| pipeline | eager | 1 | obs_1 | 1235 | 0.0 | 0.0 | True |
| stale_control | eager | 1 | obs_1 | 1235 | 0.12597131729125977 | 34.84254455566406 | False |
| pipeline | eager | 2 | obs_2 | 1236 | 0.0 | 0.0 | True |
| stale_control | eager | 2 | obs_2 | 1236 | 0.08257007598876953 | 177.81248474121094 | False |
| pipeline | eager | 3 | obs_3 | 1237 | 0.0 | 0.0 | True |

**Anything odd:** gate_note: BACK-FILLED 2026-09-26 after the run: corun.py's equivalence branch wrote no meta.json (fixed in bench/corun.py). Gate read at back-fill time, same boot and mode as the run (scratch/queues/fixup_30w.l

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
