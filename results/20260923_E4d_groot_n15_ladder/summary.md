# 20260923_E4d_groot_n15_ladder

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-24T06:26:01Z, model groot_n15, rung / mode all_compile_autotune
* **Command:** `bench/groot_phases.py --outdir results/20260923_E4d_groot_n15_ladder --version n15 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --warmup 20 --timed 50 --rung all_compile_autotune --force-gemm-autotune --tf32 on --h`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 32.2213 | 32.2457 | 32.1687 | 32.372 | 32.1603 | 32.4433 | 50 |
| vlm_backbone | 103.0304 | 103.0117 | 102.4124 | 103.6086 | 102.0786 | 104.1707 | 50 |
| vlm_prefix_excl_vision | 70.7277 | 70.766 | 70.2209 | 71.3418 | 69.8181 | 71.7857 | 50 |
| ae_total | 60.5484 | 60.6024 | 59.9149 | 61.4078 | 59.8724 | 61.4449 | 50 |
| data_processing_cpu | 58.9439 | 57.3384 | 44.8092 | 67.7327 | 44.7221 | 72.352 | 20 |
| e2e | 168.7451 | 169.1493 | 166.6578 | 172.8515 | 166.3054 | 173.6837 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
