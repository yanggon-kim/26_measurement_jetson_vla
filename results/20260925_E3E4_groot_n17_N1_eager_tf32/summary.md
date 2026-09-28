# 20260925_E3E4_groot_n17_N1_eager_tf32

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T21:16:54Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_tf32
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N1_eager_tf32 --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 1 --warmup 20 --timed 50 --rung eager_tf32 --tf32 on`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 69.856 | 69.8606 | 69.697 | 70.0689 | 69.6823 | 70.1689 | 50 |
| vlm_backbone | 156.1587 | 156.154 | 155.7683 | 156.5186 | 155.5929 | 157.1568 | 50 |
| vlm_prefix_excl_vision | 86.278 | 86.2934 | 85.9457 | 86.5882 | 85.8548 | 86.9879 | 50 |
| ae_total | 48.5247 | 48.5292 | 48.233 | 48.7964 | 48.1282 | 48.8108 | 50 |
| data_processing_cpu | 9.3994 | 9.5943 | 9.3503 | 10.1755 | 9.3473 | 12.5331 | 20 |
| e2e | 206.8893 | 206.8657 | 206.3568 | 207.2659 | 206.1937 | 207.8506 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
