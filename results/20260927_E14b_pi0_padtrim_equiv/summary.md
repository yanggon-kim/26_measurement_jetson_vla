# 20260927_E14b_pi0_padtrim_equiv

* **What:** T017 step 1 + 3: fp32 gate, compiled bf16 action difference, floor F, controls, verdicts
* **Conditions:** date 2026-09-27T23:16:28Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e14b_padtrim.py --mode verdict --outdir results/20260927_E14b_pi0_padtrim_equiv`

Headline — first rows of `verdicts.csv` (3 rows):

| variant | verdict | bitwise | MAXABS | REL | F_maxabs | F_rel | gate_REL_fp32 |
|---|---|---|---|---|---|---|---|
| nocam | PASS | 0/64 | 0.0733489990234375 | 0.0056154297406584945 | 0.2960973381996155 | 0.02445841475461925 | 1.4821020042799268e-06 |
| trimprompt | PASS | 0/64 | 0.12502914667129517 | 0.00593241809796758 | 0.2960973381996155 | 0.02445841475461925 | 2.3323841437800956e-07 |
| nopad | PASS | 0/64 | 0.04175996780395508 | 0.003399615506698086 | 0.2960973381996155 | 0.02445841475461925 | 1.147153563776931e-06 |

**Anything odd:** rung_note: {'rung': 'bf16_both', 'prep': 'vision_tower+multi_modal_projector->bf16, embed_image fp32 cast removed; 74 layernorm modules -> bf16 (own parameters only)', 'compiled': 'model.denoise_step@reduce-over; nocam_note: LeRobot's _preprocess_images builds the -1 dummy for the empty camera outside every timed region; it is removed from the list before the image encoder -- no SigLIP / projector call, no tokens, no mask; inputs_csv_note: copied from bench/fixtures/e14b/inputs.csv (fixed before any T017 run) after the verdict pass

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
