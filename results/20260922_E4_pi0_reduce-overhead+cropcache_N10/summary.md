# 20260922_E4_pi0_reduce-overhead+cropcache_N10

* **What:** E4 — task T003
* **Conditions:** date 2026-09-22T00:43:07Z, power mode NV Power Mode: MAXN, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead --variant cropcache --steps 4,10 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (1 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| e2e_wall | 530.9622549993946 | 530.8847016800428 | 529.589135300921 | 531.8501190503412 | 529.2973690011422 | 532.3591999986093 | 100 |

**Anything odd:** regions_note: Region instrumentation is OFF: bench/regions.py uses Python context managers and dynamo cannot inline contextlib.__enter__, so every region boundary becomes a graph break and an instrumented compiled ; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total; emc_hz_note: emc_hz reads 0 in these runs: `sudo -n cat /sys/kernel/debug/...` returned empty under the driver's environment. The EMC clock was 3,199,000,000 Hz in T001/T002 on the same unchanged device state.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
