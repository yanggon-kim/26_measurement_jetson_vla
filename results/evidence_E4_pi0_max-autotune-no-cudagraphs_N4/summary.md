# evidence_E4_pi0_max-autotune-no-cudagraphs_N4

* **What:** E4 — task T003
* **Conditions:** date 2026-09-22T01:36:02Z, power mode NV Power Mode: MAXN, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode max-autotune-no-cudagraphs --variant whole --steps 4 --warmup 5 --timed 10`

Headline — first rows of `latency_summary.csv` (1 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| e2e_wall | 530.1642834983795 | 530.1686841998162 | 529.8135053013539 | 530.5799541491069 | 529.6863200019288 | 530.7282169997052 | 10 |

**Anything odd:** regions_note: Region instrumentation is OFF: bench/regions.py uses Python context managers and dynamo cannot inline contextlib.__enter__, so every region boundary becomes a graph break and an instrumented compiled ; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
