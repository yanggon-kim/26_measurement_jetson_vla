# 20260927_E14b_pi0_padtrim_bytes_MAXN

* **What:** T017 2b: DRAM bytes per frame per padding variant (E12b graph-level ncu method)
* **Conditions:** date 2026-09-27T23:54:16Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `bash bench/run_ncu_e14b.sh results/20260927_E14b_pi0_padtrim_bytes_MAXN 5 (after a plain warm-up: bench/e14b_padtrim.py --mode ncu --frames 2)`

Headline — first rows of `dram_bytes.csv` (16 rows):

| variant | stage | workloads_per_frame | measured_read_GB_per_fr… | measured_write_GB_per_f… | measured_total_GB_per_f… | L2_read_requests_GB_per… | profiled_kernel_time_ms… |
|---|---|---|---|---|---|---|---|
| nocam | expert | 9.0 | 6.9227 | 1.046 | 7.9688 | 13.6891 | 91.599 |
| nocam | prefix | 1.0 | 14.7843 | 2.5201 | 17.3044 | 51.4475 | 183.067 |
| nocam | vision | 1.0 | 3.8289 | 0.5589 | 4.3878 | 10.5347 | 37.163 |
| nocam | vlm | 1.0 | 10.933 | 1.9613 | 12.8943 | 40.9136 | 145.505 |
| nopad | expert | 9.0 | 6.9069 | 0.9917 | 7.8986 | 13.6141 | 69.529 |
| nopad | prefix | 1.0 | 14.6394 | 2.4597 | 17.0991 | 49.5988 | 182.562 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; nocam_note: LeRobot's _preprocess_images builds the -1 dummy for the empty camera outside every timed region; it is removed from the list before the image encoder -- no SigLIP / projector call, no tokens, no mask; throttled: not sampled during the ncu capture (serialised replay; clocks do not bear on bytes); jetson_clocks applied before, same session as the stages run

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
