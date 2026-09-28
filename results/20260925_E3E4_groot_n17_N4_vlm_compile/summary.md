# 20260925_E3E4_groot_n17_N4_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:23:50Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N4_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 4 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --ll`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.8148 | 70.8855 | 70.6613 | 71.2619 | 70.5904 | 72.4185 | 50 |
| vlm_backbone | 155.9362 | 156.2339 | 155.523 | 158.117 | 155.3786 | 161.0633 | 50 |
| vlm_prefix_excl_vision | 85.1124 | 85.3484 | 84.8617 | 86.5137 | 84.623 | 90.0206 | 50 |
| ae_total | 79.9338 | 79.9617 | 79.8998 | 79.9857 | 79.8804 | 81.188 | 50 |
| data_processing_cpu | 9.292 | 9.3566 | 9.2434 | 9.6182 | 9.204 | 10.2616 | 20 |
| e2e | 238.0675 | 238.441 | 237.6972 | 240.5938 | 237.5068 | 243.3078 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
