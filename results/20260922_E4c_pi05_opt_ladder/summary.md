# 20260922_E4c_pi05_opt_ladder

* **What:** E4c (optimisation ladder) — task T004
* **Conditions:** date 2026-09-24T06:32:01Z, power mode NV Power Mode: MAXN, model lerobot/pi05_libero_base
* **Command:** `python bench/opt_ladder.py --outdir results/20260922_E4c_pi05_opt_ladder --steps 10 --warmup 20 --timed 10`

Headline — first rows of `opt_ladder.csv` (15 rows):

| N | rung | label | matmul_precision | compiled | regions | prep | compile_call_s |
|---|---|---|---|---|---|---|---|
| 10 | eager_fp32 | eager (matmul_precision… | highest |  | on |  |  |
| 10 | eager_tf32 | + TF32 (matmul_precisio… | high |  | on |  |  |
| 10 | ae_compile | + compile denoise_step … | high | model.denoise_step@max-… | on |  | 1.776 |
| 10 | ae_graphs | + CUDA graphs on denois… | high | model.denoise_step@redu… | on |  | 1.766 |
| 10 | vis_compile | + compile embed_image (… | high | model.denoise_step@redu… | on |  | 1.776 |
| 10 | vis_graphs | + CUDA graphs on embed_… | high | model.denoise_step@redu… | on |  | 1.771 |

**Anything odd:** notes: ['A fresh policy is loaded for every rung. torch.compile is not reversible, and a compiled callable left on the module would contaminate the rungs below it.', "Rungs 0-3 keep regions on; 4-5 cannot, b; meta_correction: {'date': '2026-09-25', 'by': 'watchdog audit', 'what': "opt_ladder.py wrote pi0's checkpoint/commit (P.CHECKPOINT/P.REVISION), fixed prefix/suffix token counts and a command without --model for EVERY 

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
