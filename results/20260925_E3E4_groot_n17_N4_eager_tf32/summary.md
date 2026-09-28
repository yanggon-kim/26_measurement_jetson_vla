# 20260925_E3E4_groot_n17_N4_eager_tf32

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T21:19:26Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_tf32
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N4_eager_tf32 --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 4 --warmup 20 --timed 50 --rung eager_tf32 --tf32 on`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.3184 | 70.3739 | 70.1063 | 70.6871 | 70.1015 | 71.3934 | 50 |
| vlm_backbone | 157.4181 | 157.5392 | 156.9541 | 158.224 | 156.82 | 160.289 | 50 |
| vlm_prefix_excl_vision | 87.0787 | 87.1652 | 86.7465 | 87.7373 | 86.4731 | 89.8266 | 50 |
| ae_total | 205.725 | 205.773 | 204.5434 | 207.1143 | 203.3532 | 208.6237 | 50 |
| data_processing_cpu | 9.3556 | 9.3945 | 9.3124 | 9.602 | 9.312 | 9.8494 | 20 |
| e2e | 365.2491 | 365.5371 | 364.1821 | 367.4284 | 364.0154 | 368.7075 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
