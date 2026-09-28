# regions_E4_pi0_reduce-overhead+step_only_N10

* **What:** E4 — task T003
* **Conditions:** date 2026-09-22T01:43:06Z, power mode NV Power Mode: MAXN, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead --variant step_only --steps 4,10 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (24 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_step[0] | 17.62566375732422 | 17.64386878967285 | 17.477122116088868 | 17.84996461868286 | 17.40470314025879 | 18.0731201171875 | 100 |
| ae_step[1] | 16.919440269470215 | 16.933207969665528 | 16.844276332855223 | 17.02206907272339 | 16.77631950378418 | 17.968063354492188 | 100 |
| ae_step[2] | 16.883440017700195 | 16.905670070648192 | 16.804989624023438 | 17.002708435058594 | 16.788991928100586 | 18.483264923095703 | 100 |
| ae_step[3] | 16.814656257629395 | 16.829678974151612 | 16.74101505279541 | 16.908372116088866 | 16.715456008911133 | 17.714271545410156 | 100 |
| ae_step[4] | 16.837519645690918 | 16.840926742553712 | 16.76777849197388 | 16.93164987564087 | 16.726720809936523 | 17.004575729370117 | 100 |
| ae_step[5] | 16.77940845489502 | 16.790703964233398 | 16.72512149810791 | 16.856258392333984 | 16.706783294677734 | 17.277311325073242 | 100 |

**Anything odd:** regions_note: Region instrumentation is ON: the compiled region is graph-broken at every region boundary. Labelled variant; not the headline compiled number.; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
