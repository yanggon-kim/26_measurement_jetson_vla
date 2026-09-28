# 20260926_E14_groot_n17_savings_MAXN

* **What:** T016 step 3 row 7: GR00T N1.7 action equivalence of keep1 / kvonce / keep1+kvonce (compiled and eager) with run-to-run controls
* **Conditions:** date 2026-09-27T20:39:19Z, power mode NV Power Mode: MAXN, rung / mode vlm_compile (compiled sets) and eager_tf32 (eager sets)
* **Command:** `bash scratch/queues/t016_groot_actions_v2.sh (groot_phases.py --save-actions 50:<dir>/actions.npz per set; python bench/e14_groot_equiv.py)`

Headline — first rows of `equiv.csv` (450 rows):

| pair | kind | inference | seed | bitwise | maxabs | rel |
|---|---|---|---|---|---|---|
| ref vs ref_rep | control | 0 | 5000 | True | 0.0 | 0.0 |
| ref vs ref_rep | control | 1 | 5001 | True | 0.0 | 0.0 |
| ref vs ref_rep | control | 2 | 5002 | True | 0.0 | 0.0 |
| ref vs ref_rep | control | 3 | 5003 | True | 0.0 | 0.0 |
| ref vs ref_rep | control | 4 | 5004 | True | 0.0 | 0.0 |
| ref vs ref_rep | control | 5 | 5005 | True | 0.0 | 0.0 |

**Anything odd:** note_compiled_vs_eager: eager vs compiled REL 1.27 equals the seed-to-seed spread (1.23): the two paths draw different noise from the same seed, so no eager/compiled floor is defined; note_recompile: torch._dynamo recompile_limit (8) warning on transformers wrapped_forward in every compiled GR00T run, including all earlier E3/E4/E7 and T016 timing runs; same configuration; throttled: not applicable to equality of outputs; runs 13:28-13:39 local under jetson_clocks

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
