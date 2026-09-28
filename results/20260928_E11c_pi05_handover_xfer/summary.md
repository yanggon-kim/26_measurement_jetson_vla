# 20260928_E11c_pi05_handover_xfer

* **What:** E11c — task T011
* **Conditions:** date 2026-09-28T20:16:13Z, power mode NV Power Mode: MAXN, model pi05
* **Command:** `python bench/xfer_exact.py --model pi05 --outdir results/20260928_E11c_pi05_handover_xfer`

Headline — first rows of `model_inputs.csv` (120 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_fp32;… | 0.19363 | ms |
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_fp32;… | 0.190392 | ms |
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:h2d | size_label=scores_fp32;… | 0.200646 | ms |
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_fp32;… | 0.274062 | ms |
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_fp32;… | 0.269084 | ms |
| pi05 | a217bfd3b14673cf2ce597e… | ace31e262cfed6fa45f3df1… | xfer_latency | pageable:d2h | size_label=scores_fp32;… | 0.295586 | ms |

**Anything odd:** notes: ["Methods and timing are T010's (bench/xfer_latency.py), unchanged; no size sweep (T011).", 'zero_copy sizes are rounded down to a multiple of 4 B (float32 elements); size_bytes records it.', "The fix

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
