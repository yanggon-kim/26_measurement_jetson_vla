# regions_E4_pi0_reduce-overhead+step_only_N4

* **What:** E4 — task T003
* **Conditions:** date 2026-09-22T01:41:47Z, power mode NV Power Mode: MAXN, model pi0 (LeRobot implementation)
* **Command:** `TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead --variant step_only --steps 4,10 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (12 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_step[0] | 17.690815925598145 | 17.703898792266845 | 17.56777753829956 | 17.845118045806885 | 17.492095947265625 | 18.192895889282227 | 100 |
| ae_step[1] | 16.9553279876709 | 16.96824703216553 | 16.89553346633911 | 17.119925022125244 | 16.860063552856445 | 17.417503356933594 | 100 |
| ae_step[2] | 16.918736457824707 | 16.924810905456543 | 16.849250411987306 | 17.010915279388428 | 16.80963134765625 | 17.23459243774414 | 100 |
| ae_step[3] | 16.84881591796875 | 16.86400297164917 | 16.78646650314331 | 16.970543670654298 | 16.74403190612793 | 17.466751098632812 | 100 |
| ae_step_individual[0] | 17.690815925598145 | 17.703898792266845 | 17.56777753829956 | 17.845118045806885 | 17.492095947265625 | 18.192895889282227 | 100 |
| ae_step_individual[1] | 16.9553279876709 | 16.96824703216553 | 16.89553346633911 | 17.119925022125244 | 16.860063552856445 | 17.417503356933594 | 100 |

**Anything odd:** regions_note: Region instrumentation is ON: the compiled region is graph-broken at every region boundary. Labelled variant; not the headline compiled number.; notes: ['ae_layer[j] instrumentation is OFF in E4 (it costs the step it lands on ~2.6 ms — T002 review finding).', 'Under CUDA graphs the per-region CUDA events may be absorbed into the graph; where ae_total

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
