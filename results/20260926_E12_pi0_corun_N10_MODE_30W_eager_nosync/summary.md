# 20260926_E12_pi0_corun_N10_MODE_30W_eager_nosync

* **What:** E12 (co-run: prefix(k) || expert(k-1) on one GPU) — task T014
* **Conditions:** power mode NV Power Mode: MODE_30W, model pi0, rung / mode eager
* **Command:** `bench/corun.py --allow-capped --mode timing --path eager --nosync --steps 10 --frames 100 --warm 20 --outdir results/20260926_E12_pi0_corun_N10_MODE_30W_eager_nosync`

Headline — first rows of `corun_ratios.csv` (1 rows):

| mechanism | power_mode | sm_count | N | A_iso_ms | B_iso_ms | T_serial_ms | A_co_ms |
|---|---|---|---|---|---|---|---|
| baseline_eager | NV Power Mode: MODE_30W | 8 | 10 | 1788.6412 | 652.912 | 2437.2821 | 1788.822 |

**Anything odd:** not_measured: ['compiled bf16_both co-run: the stock path corrupts non-deterministically when pipelined across two streams (max|d| 1.273 on one run, all-NaN on another) while compiled SERIAL is correct and eager PI; notes: ['NO OVERLAP OCCURRED. T_pipe equals A_iso + B_iso, and host issue/wall = 1.0000 -- the CPU spends the entire wall clock issuing kernels, so the two streams never both have resident work. s and p ther; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
