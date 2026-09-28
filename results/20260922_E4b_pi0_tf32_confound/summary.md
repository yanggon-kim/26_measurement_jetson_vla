# 20260922_E4b_pi0_tf32_confound

* **What:** E4b (TF32 confound) — task T004
* **Conditions:** date 2026-09-22T17:32:52Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/tf32_confound.py --outdir results/20260922_E4b_pi0_tf32_confound --seconds 20.0 --steps 4`

Headline — first rows of `tf32_confound.csv` (6 rows):

| phase | matmul_precision | compiled | median_ms | mean_ms | p5_ms | p95_ms | min_ms |
|---|---|---|---|---|---|---|---|
| prefix | highest | False | 478.008013 | 478.296253 | 477.718426 | 479.171365 | 477.685296 |
| ae | highest | False | 232.75915 | 233.097751 | 232.127384 | 235.113058 | 231.972589 |
| full | highest | False | 716.248864 | 716.362311 | 715.439817 | 717.399158 | 715.414212 |
| prefix | high | False | 341.435527 | 341.23066 | 340.450187 | 341.926246 | 340.382298 |
| ae | high | False | 230.112511 | 230.264037 | 229.686042 | 231.047147 | 229.482182 |
| full | high | False | 575.717343 | 575.872299 | 574.990727 | 577.041764 | 574.380638 |

**Anything odd:** notes: ['The policy is loaded WITHOUT compilation in both halves. Only torch.set_float32_matmul_precision changes between them.', '`prefix` and `ae` call prefix_once / denoise_step directly and never enter s; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
