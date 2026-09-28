# 20260925_E3E4_groot_n17_N8_eager_tf32

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T21:22:27Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_tf32
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N8_eager_tf32 --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 8 --warmup 20 --timed 50 --rung eager_tf32 --tf32 on`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.2513 | 70.2577 | 70.0919 | 70.3909 | 70.0712 | 70.8777 | 50 |
| vlm_backbone | 158.4558 | 158.5052 | 157.9544 | 159.2914 | 157.8642 | 159.477 | 50 |
| vlm_prefix_excl_vision | 88.2194 | 88.2475 | 87.7779 | 88.8542 | 87.596 | 89.1313 | 50 |
| ae_total | 420.0358 | 420.163 | 418.5158 | 421.8582 | 418.269 | 423.664 | 50 |
| data_processing_cpu | 9.4014 | 9.4852 | 9.3555 | 10.0276 | 9.3543 | 10.4213 | 20 |
| e2e | 580.6299 | 580.85 | 579.1232 | 583.2885 | 578.2961 | 584.6904 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
