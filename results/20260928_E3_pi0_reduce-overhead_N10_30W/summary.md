# 20260928_E3_pi0_reduce-overhead_N10_30W

* **What:** E4 — task T003
* **Conditions:** date 2026-09-28T23:58:19Z, power mode NV Power Mode: MODE_30W, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead --variant whole --steps 4,10 --warmup 20 --timed 100 --allow-capped --exp E3 --suffix 30W`

Headline — first rows of `latency_summary.csv` (1 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| e2e_wall | 2061.5058695000243 | 2061.182678329992 | 2058.433546399897 | 2062.7308073001814 | 2057.412267000018 | 2065.2228070000547 | 100 |

**Anything odd:** regions_note: Region instrumentation is OFF: bench/regions.py uses Python context managers and dynamo cannot inline contextlib.__enter__, so every region boundary becomes a graph break and an instrumented compiled ; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total; backfill_note: 2026-09-28: null inductor_sm_gate_patched replaced by its meaning (writer fixed); the run log copied into raw/

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
