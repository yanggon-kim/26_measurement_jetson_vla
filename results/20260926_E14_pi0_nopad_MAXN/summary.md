# 20260926_E14_pi0_nopad_MAXN

* **What:** T016 step 3 (A4): pi0_bf16_both+nopad vs pi0_bf16_both (pad) -- DRAM bytes per frame, E12b graph-level ncu method
* **Conditions:** date 2026-09-27T14:57:47Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `bash bench/run_ncu_e14.sh results/20260926_E14_pi0_nopad_MAXN/bytes pad,nopad 5 (after a no-ncu warm-up: bench/e14_bytes_target.py --variants pad,nopad --frames 2)`
* **Sub-runs:** bytes, equiv, time

Headline — first rows of `bytes/dram_bytes.csv` (4 rows):

| variant | stage | workloads_per_frame | measured_read_GB_per_fr… | measured_write_GB_per_f… | measured_total_GB_per_f… | L2_read_requests_GB_per… | profiled_kernel_time_ms… |
|---|---|---|---|---|---|---|---|
| nopad | expert | 9.0 | 6.9069 | 0.9917 | 7.8986 | 13.6146 | 69.408 |
| nopad | prefix | 1.0 | 14.6444 | 2.4594 | 17.1038 | 49.6004 | 182.55 |
| pad | expert | 9.0 | 7.0298 | 1.2581 | 8.2879 | 15.0145 | 105.266 |
| pad | prefix | 1.0 | 17.6535 | 3.8637 | 21.5172 | 73.9847 | 252.177 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; throttled: not sampled during the ncu capture (ncu serialises kernels, so clocks do not bear on bytes); jetson_clocks applied immediately before (queue scratch/queues/t016_followup.sh)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
