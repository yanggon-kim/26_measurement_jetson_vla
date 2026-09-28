# 20260925_E3E4_groot_n17_N8_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:34:57Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N8_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 8 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --ll`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.3476 | 70.3579 | 70.1639 | 70.6333 | 70.1482 | 70.7175 | 50 |
| vlm_backbone | 154.6402 | 154.6899 | 154.292 | 155.2453 | 154.0888 | 155.6283 | 50 |
| vlm_prefix_excl_vision | 84.2959 | 84.332 | 83.955 | 84.6991 | 83.8632 | 84.9712 | 50 |
| ae_total | 152.0495 | 152.2733 | 151.9786 | 153.0268 | 151.9345 | 153.0684 | 50 |
| data_processing_cpu | 9.3254 | 9.7632 | 9.2457 | 10.8074 | 9.223 | 16.4366 | 20 |
| e2e | 309.0976 | 309.1995 | 308.567 | 310.1466 | 308.2841 | 310.3935 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
