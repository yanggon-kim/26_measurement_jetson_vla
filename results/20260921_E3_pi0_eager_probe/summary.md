# 20260921_E3_pi0_eager_probe

* **What:** E3 — task T002
* **Conditions:** date 2026-09-21T22:54:14Z, power mode NV Power Mode: MAXN, model pi0 (LeRobot implementation)
* **Command:** `python bench/probe_e3.py --outdir results/20260921_E3_pi0_eager_probe --warmup 20 --timed 100`

Headline — first rows of `kv_deepcopy.csv` (100 rows):

| iter | ms |
|---|---|
| 0 | 5.732489999900281 |
| 1 | 5.708206000235805 |
| 2 | 5.688815999747021 |
| 3 | 5.713452999771107 |
| 4 | 5.670737999935227 |
| 5 | 5.692528000508901 |

**Anything odd:** notes: ['Probes run after the main sweep to close three gaps an independent review found: the step-0 offset was confounded with ae_layer instrumentation, the per-step KV deepcopy was missed, and the first pe; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
