# 20260922_E11a_pi0_hostseg

* **What:** E11a — task T010
* **Conditions:** date 2026-09-22T07:01:59Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/host_segments.py --outdir results/20260922_E11a_pi0_hostseg --warmup 20 --timed 1000`

Headline — first rows of `model_inputs.csv` (1002 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=0 | 0.392714996 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=0 | 0.384937398 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=0 | 0.415854199 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=2 | 0.399307501 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=2 | 0.392167801 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_host_segment | rmsnorm1 | layer=0;gap_ms=2 | 0.498823599 | ms |

**Anything odd:** not_measured: ["suffix_embed exec_style=cudagraph: AcceleratorError: CUDA error: operation failed due to a previous error during capture\nSearch for `cudaErrorStreamCaptureInvalidated' in https://docs.nvidia.com/cu; notes: ['Segments are the DENOISING path: denoise_step -> paligemma_with_expert.forward with inputs_embeds=[None, suffix] -> gemma_expert.model.forward -> the decoder layer at pi_gemma.py:157-188. compute_la; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
