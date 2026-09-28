# 20260928_E11c_smolvla_handover_xfer

* **What:** E11c — task T011
* **Conditions:** date 2026-09-28T20:16:23Z, power mode NV Power Mode: MAXN, model smolvla
* **Command:** `python bench/xfer_exact.py --model smolvla --outdir results/20260928_E11c_smolvla_handover_xfer`

Headline — first rows of `model_inputs.csv` (168 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.134911 | ms |
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.132668 | ms |
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.139139 | ms |
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.148543 | ms |
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.14562 | ms |
| smolvla | d9f33c94a60fb382c90dea2… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.152646 | ms |

**Anything odd:** notes: ["Methods and timing are T010's (bench/xfer_latency.py), unchanged; no size sweep (T011).", 'zero_copy sizes are rounded down to a multiple of 4 B (float32 elements); size_bytes records it.', "The fix

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
