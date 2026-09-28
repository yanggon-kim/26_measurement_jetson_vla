# 20260922_E11f_pi0_pipelining

* **What:** E11f — task T010
* **Conditions:** date 2026-09-22T10:55:09Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/pipelining.py --outdir results/20260922_E11f_pi0_pipelining`

Headline — first rows of `model_inputs.csv` (14 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | prefix_throughput |  | mode=eager | 2.1007 | inf_per_s |
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | ae_throughput |  | mode=eager | 4.2934 | inf_per_s |
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | e2e_throughput |  | mode=eager | 1.4021 | inf_per_s |
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | prefix_throughput |  | mode=reduce-overhead | 2.9373 | inf_per_s |
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | ae_throughput |  | mode=reduce-overhead | 4.3526 | inf_per_s |
| pi0 | 45dcc8fc0e02601c8ccf055… | 5eeddd5cacf36a5073d9ff3… | e2e_throughput |  | mode=reduce-overhead | 2.759 | inf_per_s |

**Anything odd:** notes: ["gpu_idle_ae_total is region time minus the UNION of kernel intervals inside the ae_step NVTX spans, so overlapping kernels are not double-counted. It is the GPU time already free for the next frame'

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
