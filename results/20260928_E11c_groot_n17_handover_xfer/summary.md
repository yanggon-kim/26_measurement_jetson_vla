# 20260928_E11c_groot_n17_handover_xfer

* **What:** E11c — task T011
* **Conditions:** date 2026-09-28T20:28:18Z, power mode NV Power Mode: MAXN, model groot_n17
* **Command:** `python bench/xfer_exact.py --sizes-json results/20260928_E11c_groot_n17_handover/xfer_sizes.json --model-name groot_n17 --outdir results/20260928_E11c_groot_n17_handover_xfer`

Headline — first rows of `model_inputs.csv` (192 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.087228 | ms |
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.083512 | ms |
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:h2d | size_label=scores_self_… | 0.091144 | ms |
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.093949 | ms |
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.091291 | ms |
| groot_n17 | 376ba890cff8c9de64d71d9… | 4110b465e5152be1a203d44… | xfer_latency | pageable:d2h | size_label=scores_self_… | 0.097012 | ms |

**Anything odd:** notes: ["Methods and timing are T010's (bench/xfer_latency.py), unchanged; no size sweep (T011).", 'zero_copy sizes are rounded down to a multiple of 4 B (float32 elements); size_bytes records it.', "The fix

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
