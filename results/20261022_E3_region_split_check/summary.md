# E3 region split check (T004 step 2, protocol §2) — 2026-10-22, MAXN, jetson_clocks, gate passed per model, eager_fp32

One nsys capture per model (2 inferences, regions installed by `bench/regions.py`, `bench/nsys_regions_target.py`);
`bench/region_split_check.py` assigns every kernel of `cuda_gpu_trace` to the GPU-projected NVTX intervals.

| model | vision ranges / kernels | vlm_prefix ranges / kernels | kernels in both | intervals overlap | vision + prefix ÷ backbone (E3, N = 4, median of 100) |
|---|---|---|---|---|---|
| π0 | 6 / 2,328 | 2 / 1,770 | 0 | no | n/a (no enclosing call) |
| π0.5 | 6 / 2,328 | 2 / 1,578 | 0 | no | n/a |
| SmolVLA | 4 / 864 | 2 / 2,414 | 0 | no | n/a |
| GR00T N1.7 | 2 / 1,596 | 2 / 2,448 | 0 | no | **0.895: fails the 2 % rule; cause identified below** |
| GR00T N1.6 | 2 / 1,018 | 2 / 1,868 | 0 | no | 0.993: passes |
| GR00T N1.5 | 2 / 1,370 | 2 / 1,410 | 0 | no | 0.995: passes |

**GR00T N1.7's gap.** In E3, `vlm_backbone` − `vision` − `vlm_prefix` = 6.6 ms. In this capture, 110 kernels (8.1 ms)
over 2 inferences run inside `backbone.forward` but outside both regions.
- **One GEMM per inference** (`nvjet_sm110_tst_128x144…`, grid 2376, 3.63 / 3.95 ms) starts as `language_model`
  returns. It is the **`lm_head` vocabulary projection**: 273 tokens × 2,048 × 151,936 = 169.9 GFLOP, ≈ 47 TFLOP/s.
  `Qwen3VLForConditionalGeneration.forward` computes logits (transformers `modeling_qwen3_vl.py:1362`), but GR00T's
  `Qwen3Backbone.forward` keeps only `outputs.hidden_states[-1]` (`gr00t/model/modules/qwen3_backbone.py`). **The
  logits are computed and never used.**
- **Index / scatter kernels** (`index_elementwise`, `masked_scatter`, cub select / scan): 0.2 ms per inference.
- **The rest** is CPU-side multimodal position-id work (`get_rope_index`) with the GPU idle.

The regions stay as defined (vision = `get_image_features`, vlm_prefix = `language_model`). The gap is reported as
part of "other" and named here. The protocol says to report stack properties, not fix them; removing the logits would
be a labelled variant.

LeRobot models have no enclosing call around vision and prefix (`embed_prefix` runs `embed_image` per camera, then
the prefill `forward`), so only the disjointness check applies to them. nsys binaries are not committed
(`results/20261022_E0_secrets_test/`); the `nsys stats` CSV exports are (zstd above 1 MB).
