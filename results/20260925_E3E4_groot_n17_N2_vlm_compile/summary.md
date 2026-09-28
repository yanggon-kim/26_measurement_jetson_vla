# 20260925_E3E4_groot_n17_N2_vlm_compile

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T23:22:22Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode vlm_compile
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N2_vlm_compile --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 2 --warmup 20 --timed 50 --rung vlm_compile --tf32 on --head-mode reduce-overhead --ll`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 70.0926 | 70.1005 | 69.9667 | 70.2684 | 69.9437 | 70.2797 | 50 |
| vlm_backbone | 154.2207 | 154.2134 | 153.7563 | 154.7073 | 153.7222 | 154.9299 | 50 |
| vlm_prefix_excl_vision | 84.1115 | 84.1129 | 83.7308 | 84.5354 | 83.6586 | 84.7786 | 50 |
| ae_total | 43.5527 | 43.6389 | 43.5227 | 43.9251 | 43.5154 | 43.9346 | 50 |
| data_processing_cpu | 9.2695 | 10.2237 | 9.193 | 15.9982 | 9.1924 | 16.2194 | 20 |
| e2e | 200.0323 | 200.0153 | 199.4607 | 200.5824 | 199.4465 | 200.7576 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
