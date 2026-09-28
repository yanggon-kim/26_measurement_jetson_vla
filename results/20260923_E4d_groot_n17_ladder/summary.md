# 20260923_E4d_groot_n17_ladder

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-23T15:18:57Z, model groot_n17, rung / mode vlm_graphs
* **Command:** `bench/groot_phases.py --outdir results/20260923_E4d_groot_n17_ladder --checkpoint nvidia/GR00T-N1.7-3B --version n17 --warmup 20 --timed 50 --rung vlm_graphs --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhe`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 71.3938 | 74.2499 | 70.5605 | 83.5017 | 70.4879 | 117.5004 | 50 |
| vlm_backbone | 161.0919 | 168.7733 | 157.6267 | 197.8706 | 156.8489 | 260.8481 | 50 |
| vlm_prefix_excl_vision | 89.2466 | 94.5234 | 86.4365 | 123.2193 | 86.1509 | 187.1715 | 50 |
| ae_total | 80.2564 | 80.3199 | 80.1253 | 80.5176 | 80.0507 | 82.688 | 50 |
| data_processing_cpu | 37.7198 | 40.3937 | 31.0143 | 60.9585 | 27.0245 | 72.5031 | 20 |
| e2e | 244.1532 | 251.9188 | 240.5886 | 280.9203 | 239.6038 | 343.9942 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
