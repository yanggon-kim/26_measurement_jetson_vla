# 20260922_E11b_xfer

* **What:** E11b — task T010
* **Conditions:** date 2026-09-22T07:04:08Z, power mode NV Power Mode: MAXN
* **Command:** `python bench/xfer_latency.py --outdir results/20260922_E11b_xfer`

Headline — first rows of `model_inputs.csv` (243 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:h2d | size_bytes=4096 | 0.058241 | ms |
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:h2d | size_bytes=4096 | 0.055169 | ms |
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:h2d | size_bytes=4096 | 0.06544 | ms |
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:d2h | size_bytes=4096 | 0.061889 | ms |
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:d2h | size_bytes=4096 | 0.059041 | ms |
| none |  | a3878f563ddabe86dc71d82… | xfer_latency | pageable:d2h | size_bytes=4096 | 0.064864 | ms |

**Anything odd:** notes: ['MB = 10^6 B for fit_b (§13.3); KiB/MiB labels are binary.', 'The fixed term fit_a is the quantity this experiment exists for: the hybrid model assumes a = 0 and prices transfers purely as bytes / 14; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
