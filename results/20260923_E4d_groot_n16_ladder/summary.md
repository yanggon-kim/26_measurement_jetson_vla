# 20260923_E4d_groot_n16_ladder

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-24T06:19:21Z, model groot_n16, rung / mode all_compile_autotune
* **Command:** `bench/groot_phases.py --outdir results/20260923_E4d_groot_n16_ladder --version n16 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --warmup 20 --timed 50 --rung all_compile_autotune --force-gemm-autotune --tf32 on --h`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 48.8015 | 52.4646 | 46.6904 | 59.6841 | 46.6511 | 60.3204 | 50 |
| vlm_backbone | 99.7264 | 106.2392 | 89.592 | 125.2643 | 89.2468 | 126.604 | 50 |
| vlm_prefix_excl_vision | 52.6145 | 53.7745 | 42.8195 | 66.1676 | 42.5606 | 67.4361 | 50 |
| ae_total | 72.8331 | 72.8348 | 72.7296 | 72.9148 | 72.7008 | 73.1341 | 50 |
| data_processing_cpu | 45.4215 | 46.3704 | 37.6079 | 52.8246 | 31.4651 | 84.4025 | 20 |
| e2e | 174.5869 | 181.1955 | 164.5799 | 200.2498 | 164.2955 | 201.638 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
