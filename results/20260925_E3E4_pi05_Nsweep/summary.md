# 20260925_E3E4_pi05_Nsweep

* **What:** E4c (optimisation ladder) — task T004
* **Conditions:** date 2026-09-25T21:04:00Z, power mode NV Power Mode: MAXN, model lerobot/pi05_libero_base
* **Command:** `python bench/opt_ladder.py --model pi05 --steps 1,2,4,6,8,10 --rungs eager_tf32,bf16_both_graphs --outdir results/20260925_E3E4_pi05_Nsweep  (reconstructed from bench/queues/queue_prereboot.sh stage e3e4_sweep)`

Headline — first rows of `opt_ladder.csv` (12 rows):

| N | rung | label | matmul_precision | status | not_applicable_reason | compiled | regions |
|---|---|---|---|---|---|---|---|
| 1 | eager_tf32 | + TF32 (matmul_precisio… | high | measured |  |  | on |
| 1 | bf16_both_graphs | B2+B3 with CUDA graphs … | high | measured |  | model.denoise_step@redu… | on |
| 2 | eager_tf32 | + TF32 (matmul_precisio… | high | measured |  |  | on |
| 2 | bf16_both_graphs | B2+B3 with CUDA graphs … | high | measured |  | model.denoise_step@redu… | on |
| 4 | eager_tf32 | + TF32 (matmul_precisio… | high | measured |  |  | on |
| 4 | bf16_both_graphs | B2+B3 with CUDA graphs … | high | measured |  | model.denoise_step@redu… | on |

**Anything odd:** notes: ['A fresh policy is loaded for every rung. torch.compile is not reversible, and a compiled callable left on the module would contaminate the rungs below it.', "Rungs 0-3 keep regions on; 4-5 cannot, b; meta_correction: {'date': '2026-09-25', 'by': 'watchdog audit', 'what': "opt_ladder.py wrote pi0's checkpoint/commit (P.CHECKPOINT/P.REVISION), fixed prefix/suffix token counts and a command without --model for EVERY 

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
