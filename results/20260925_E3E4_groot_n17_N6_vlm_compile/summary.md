# 20260925_E3E4_groot_n17_N6_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:28:37Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N6_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 6 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --ll`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.2765 | 70.6605 | 70.0585 | 73.3417 | 69.9997 | 77.1892 | 50 |
| vlm_backbone | 154.7091 | 156.2216 | 153.9884 | 162.7358 | 153.8573 | 179.0011 | 50 |
| vlm_prefix_excl_vision | 84.5206 | 85.5611 | 83.8524 | 90.7885 | 83.8206 | 101.8119 | 50 |
| ae_total | 116.4768 | 116.5393 | 116.3781 | 116.8195 | 116.3681 | 118.4225 | 50 |
| data_processing_cpu | 9.2221 | 9.2485 | 9.1753 | 9.3789 | 9.1655 | 9.527 | 20 |
| e2e | 273.3694 | 275.0742 | 272.6627 | 281.8979 | 272.5027 | 298.7662 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
