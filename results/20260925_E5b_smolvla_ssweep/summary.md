# 20260925_E5b_smolvla_ssweep

* **What:** E5b S sweep (T008 step 6; docs/03 'S sweep (isolated kernels)')
* **Conditions:** power mode NV Power Mode: MAXN, model smolvla

Headline — first rows of `s_sweep.csv` (20 rows):

| model | proj | proj_name | in | out | dtype | S | matmul_precision |
|---|---|---|---|---|---|---|---|
| smolvla | largest | layers[0].mlp.gate_proj | 720 | 2048 | bfloat16 | 1 | high |
| smolvla | largest | layers[0].mlp.gate_proj | 720 | 2048 | bfloat16 | 16 | high |
| smolvla | largest | layers[0].mlp.gate_proj | 720 | 2048 | bfloat16 | 50 | high |
| smolvla | largest | layers[0].mlp.gate_proj | 720 | 2048 | bfloat16 | 256 | high |
| smolvla | q | layers[0].self_attn.q_p… | 720 | 960 | bfloat16 | 1 | high |
| smolvla | q | layers[0].self_attn.q_p… | 720 | 960 | bfloat16 | 16 | high |

**Anything odd:** S_note: 50 is T008's stated S; the loaded config says 50; timing_correction: CUDA-event column (s_sweep.csv median_us) is SUPERSEDED: it timed each call with its own event pair + synchronize, adding ~45-75 us on Orin. Kernel time per call is in s_sweep_ncu.csv (bench/sweep_ncu; gate_note: read when this meta.json was written (after the run); the stage's own gate() refused non-MAXN before it started

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
