# GR00T region split check, v2 (T004 step 2, protocol §2) — 2026-10-22, MAXN, jetson_clocks, gate passed, eager_fp32

Regions as defined after the T004 review (`bench/regions.py`, the same in all three versions):
- `vision` = the image encoder with its projector;
- `vlm_prefix` = the LLM **decoder stack**;
- `backbone_lm_head` = the vocabulary projection the shipped code computes and GR00T discards.

One nsys capture per model (2 inferences); sums from the E3 v2 runs at N = 4 (median of 100 iterations, CUDA events).

| model | kernels in vision / prefix / both | (vision + prefix) ÷ backbone | (vision + prefix + lm_head) ÷ backbone | unused lm_head ms | rest of the backbone: kernels, ms over 2 inferences |
|---|---|---|---|---|---|
| GR00T N1.7 | 1,596 / 2,446 / 0 | 0.895 | **0.953** | 3.46 | 108 kernels, 0.51 ms (+ CPU-side `get_rope_index`) |
| GR00T N1.6 | 1,018 / 1,866 / 0 | 0.907 | **0.992** | 3.92 | 28 kernels, 0.13 ms |
| GR00T N1.5 | 1,370 / 1,408 / 0 | 0.712 | **0.995** | 19.88 | 24 kernels, 0.22 ms |

- Vision and prefix kernel sets are disjoint in all three, and the intervals never overlap.
- With the unused `lm_head` named as its own term, N1.6 and N1.5 sum to the backbone within 2 %.
- N1.7 leaves 4.7 % (≈ 3 ms) unaccounted. Its GPU part is 0.25 ms per inference of index / scatter kernels; the rest
  is CPU-side multimodal position-id work (`get_rope_index`) with the GPU idle. Named, not fixed (protocol §2).
- **The unused `lm_head` is real cost in the shipped code:** 3.5 / 3.9 / 19.9 ms per inference (N1.7 / N1.6 / N1.5),
  i.e. 2–14 % of e2e. It belongs to no part of the action pipeline. It is reported in "other".

Supersedes the GR00T rows of `results/20261022_E3_region_split_check/` (different region definitions per version).
