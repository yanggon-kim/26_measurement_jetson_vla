# 20260925_E3E4_groot_n17_N10_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:42:42Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N10_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 10 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.2681 | 70.321 | 70.0849 | 70.691 | 69.9343 | 70.7439 | 50 |
| vlm_backbone | 154.7151 | 154.8887 | 154.2721 | 155.9082 | 154.1476 | 156.2424 | 50 |
| vlm_prefix_excl_vision | 84.478 | 84.5677 | 84.0332 | 85.379 | 83.9636 | 85.5413 | 50 |
| ae_total | 188.707 | 188.9214 | 188.6363 | 189.5904 | 188.6262 | 189.7957 | 50 |
| data_processing_cpu | 9.3628 | 9.5342 | 9.3065 | 10.0214 | 9.2923 | 11.4381 | 20 |
| e2e | 345.7968 | 345.973 | 345.1109 | 347.1603 | 344.9033 | 347.4841 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
