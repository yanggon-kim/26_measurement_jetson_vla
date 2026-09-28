# 20260925_E5b_pi05_ssweep

* **What:** E5b S sweep (T008 step 6; docs/03 'S sweep (isolated kernels)')
* **Conditions:** power mode NV Power Mode: MAXN, model pi05

Headline — first rows of `s_sweep.csv` (16 rows):

| model | proj | proj_name | in | out | dtype | S | matmul_precision |
|---|---|---|---|---|---|---|---|
| pi05 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | float32 | 1 | high |
| pi05 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | float32 | 16 | high |
| pi05 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | float32 | 50 | high |
| pi05 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | float32 | 256 | high |
| pi05 | q | layers[0].self_attn.q_p… | 1024 | 2048 | float32 | 1 | high |
| pi05 | q | layers[0].self_attn.q_p… | 1024 | 2048 | float32 | 16 | high |

**Anything odd:** S_note: 50 is T008's stated S; the loaded config says 50; timing_correction: CUDA-event column (s_sweep.csv median_us) is SUPERSEDED: it timed each call with its own event pair + synchronize, adding ~45-75 us on Orin. Kernel time per call is in s_sweep_ncu.csv (bench/sweep_ncu; gate_note: read when this meta.json was written (after the run); the stage's own gate() refused non-MAXN before it started

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
