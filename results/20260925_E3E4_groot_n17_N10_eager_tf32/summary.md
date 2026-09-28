# 20260925_E3E4_groot_n17_N10_eager_tf32

* **What:** E4d (GR00T phase split)
* **Conditions:** date 2026-09-25T21:24:07Z, power mode NV Power Mode: MAXN, model groot_n17, rung / mode eager_tf32
* **Command:** `bench/groot_phases.py --outdir results/20260925_E3E4_groot_n17_N10_eager_tf32 --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 10 --warmup 20 --timed 50 --rung eager_tf32 --tf32 on`

Headline — first rows of `phase_split.csv` (6 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| vision | 69.8019 | 69.7954 | 69.6123 | 69.9829 | 69.4943 | 70.1559 | 50 |
| vlm_backbone | 154.8184 | 154.8401 | 154.2364 | 155.298 | 154.1984 | 155.5265 | 50 |
| vlm_prefix_excl_vision | 85.0333 | 85.0447 | 84.6036 | 85.4129 | 84.5029 | 85.6874 | 50 |
| ae_total | 511.0105 | 510.9811 | 509.7974 | 512.2673 | 509.7479 | 512.82 | 50 |
| data_processing_cpu | 9.4674 | 9.7535 | 9.3766 | 10.1846 | 9.3749 | 14.5766 | 20 |
| e2e | 667.9745 | 668.0458 | 666.7624 | 669.6052 | 666.5782 | 670.3315 | 50 |

**Anything odd:** fixture_note: keys and shapes are read from the checkpoint's own processor_config.json and statistics.json; pixel and state VALUES are random, which latency does not depend on; data_processing_note: data_processing_cpu is the checkpoint's own processor + collator, timed separately; e2e below is the GPU path only (model.get_action on already-collated inputs), the same split NVIDIA's benchmark_infe; vision_note: SUPERSEDED 2026-09-23: vision IS separated now. GR00T's backbone is Qwen3VLForConditionalGeneration, whose .visual and .language_model are the same two parts the LeRobot models expose as embed_image a

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
