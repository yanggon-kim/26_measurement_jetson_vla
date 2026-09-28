# 20260925_E3E4_groot_n17_N1_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:21:12Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N1_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 1 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --ll`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.0999 | 70.1487 | 69.92 | 70.5553 | 69.8612 | 70.7334 | 50 |
| vlm_backbone | 154.2579 | 154.358 | 153.8362 | 155.2902 | 153.6457 | 155.6039 | 50 |
| vlm_prefix_excl_vision | 84.1578 | 84.2093 | 83.786 | 84.7237 | 83.599 | 85.1932 | 50 |
| ae_total | 25.4101 | 25.5193 | 25.3839 | 25.92 | 25.3804 | 25.9324 | 50 |
| data_processing_cpu | 9.6182 | 11.9615 | 9.243 | 16.8497 | 9.241 | 17.06 | 20 |
| e2e | 182.074 | 182.175 | 181.6271 | 183.2179 | 181.304 | 183.5256 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
