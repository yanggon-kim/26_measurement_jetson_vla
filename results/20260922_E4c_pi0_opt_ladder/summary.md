# 20260922_E4c_pi0_opt_ladder

* **What:** E4c (optimisation ladder) — task T004
* **Conditions:** date 2026-09-23T14:18:30Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/opt_ladder.py --outdir results/20260922_E4c_pi0_opt_ladder --steps 10 --warmup 20 --timed 10`

Headline — first rows of `opt_ladder.csv` (23 rows):

| N | rung | label | matmul_precision | compiled | regions | compile_call_s | e2e_median_ms |
|---|---|---|---|---|---|---|---|
| 4 | eager_fp32 | eager (matmul_precision… | highest |  | on |  | 711.7533 |
| 4 | eager_tf32 | + TF32 (matmul_precisio… | high |  | on |  | 575.2828 |
| 4 | ae_compile | + compile denoise_step … | high | model.denoise_step@max-… | on | 2.373 | 444.3497 |
| 4 | ae_graphs | + CUDA graphs on denois… | high | model.denoise_step@redu… | on | 0.005 | 416.8519 |
| 4 | vis_compile | + compile embed_image (… | high | model.denoise_step@redu… | on | 0.009 | 413.6286 |
| 4 | vis_graphs | + CUDA graphs on embed_… | high | model.denoise_step@redu… | on | 0.009 | 413.735 |

**Anything odd:** notes: ['A fresh policy is loaded for every rung. torch.compile is not reversible, and a compiled callable left on the module would contaminate the rungs below it.', "Rungs 0-3 keep regions on; 4-5 cannot, b

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
