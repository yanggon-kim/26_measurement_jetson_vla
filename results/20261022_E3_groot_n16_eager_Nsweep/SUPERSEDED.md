Superseded by `results/20261022_E3_groot_n16_eager_Nsweep_v2/` (T004 review):
- **vlm_prefix definition.** For N1.5 / N1.6 the first version wrapped the Qwen3ForCausalLM, so vlm_prefix included the
  unused vocabulary lm_head (N1.5 ≈ 20 ms, N1.6 ≈ 4 ms). For N1.7 the same work fell in "other". v2 times the decoder
  stack as vlm_prefix in all three versions, and the lm_head as its own region `backbone_lm_head`.
- **Floors.** The first version counted all 32 embodiments of GR00T's CategorySpecificLinear weights; only the used
  category's slice is read.
- **Timings.** Vision and expert are unaffected. For N1.7, see `results/20261022_E3_groot_n17_state_vs_latency/` for
  its run-to-run expert variation.
