# 20260925_E3E4_groot_n17_N6_eager_tf32

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T21:20:52Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_tf32
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N6_eager_tf32 --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 6 --warmup 20 --timed 50 --rung eager_tf32 --tf32 on`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.0355 | 70.0305 | 69.8726 | 70.2257 | 69.8297 | 70.4856 | 50 |
| vlm_backbone | 155.3743 | 155.3994 | 154.9418 | 155.9361 | 154.8842 | 156.4804 | 50 |
| vlm_prefix_excl_vision | 85.3453 | 85.3689 | 84.9553 | 85.8189 | 84.9066 | 86.5093 | 50 |
| ae_total | 304.9734 | 305.0335 | 303.9883 | 306.139 | 303.1946 | 306.3502 | 50 |
| data_processing_cpu | 9.4088 | 9.6304 | 9.3493 | 9.7685 | 9.3476 | 13.6079 | 20 |
| e2e | 462.5855 | 462.661 | 461.7793 | 463.8768 | 461.5158 | 464.558 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
