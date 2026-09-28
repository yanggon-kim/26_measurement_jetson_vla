# 20260928_E5_ncu_smolvla_best_aelayer

* **What:** E5 C1 ncu, action-expert layer ranges (T008 step 3) — task T008
* **Conditions:** date 2026-09-28, power mode NV Power Mode: MAXN, model smolvla, rung / mode best
* **Command:** `bash bench/run_ncu_c1.sh <model> <mode> <region> <outdir> (exact ncu command in ncu/<tag>.sh)`

**Anything odd:** not_measured: ['ae_layer_0: NOT MEASURED -- the target died before any kernel was profiled: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: Fil; notes: ['Completeness audit 2026-09-28: T008 step 3 asked for ae_layer ncu captures (pi05 layer 0; smolvla layer 0 = self-attention, layer 1 = cross-attention), eager and best; they had never been run.', 'Fi

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
