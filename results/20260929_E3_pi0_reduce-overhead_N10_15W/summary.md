# 20260929_E3_pi0_reduce-overhead_N10_15W

* **What:** E4 — task T003
* **Conditions:** date 2026-09-29T00:39:39Z, power mode NV Power Mode: MODE_15W, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead --variant whole --steps 4,10 --warmup 20 --timed 100 --allow-capped --exp E3 --suffix 15W`

Headline — first rows of `latency_summary.csv` (1 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| e2e_wall | 3765.8922450001455 | 3766.4194375999978 | 3764.4939536000493 | 3768.4466629500434 | 3763.568872000178 | 3787.8372670002136 | 100 |

**Anything odd:** regions_note: Region instrumentation is OFF: bench/regions.py uses Python context managers and dynamo cannot inline contextlib.__enter__, so every region boundary becomes a graph break and an instrumented compiled ; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
