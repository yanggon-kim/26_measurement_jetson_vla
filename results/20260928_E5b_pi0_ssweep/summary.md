# 20260928_E5b_pi0_ssweep

* **What:** E5b S sweep (T007 step 3; docs/03 'S sweep (isolated kernels)') — task T007
* **Conditions:** date 2026-09-28T19:49:00Z, power mode NV Power Mode: MAXN, model pi0

Headline — first rows of `s_sweep.csv` (8 rows):

| model | proj | proj_name | in | out | dtype | S | matmul_precision |
|---|---|---|---|---|---|---|---|
| pi0 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | bfloat16 | 1 | high |
| pi0 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | bfloat16 | 16 | high |
| pi0 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | bfloat16 | 51 | high |
| pi0 | largest | layers[0].mlp.gate_proj | 1024 | 4096 | bfloat16 | 256 | high |
| pi0 | q | layers[0].self_attn.q_p… | 1024 | 2048 | bfloat16 | 1 | high |
| pi0 | q | layers[0].self_attn.q_p… | 1024 | 2048 | bfloat16 | 16 | high |

**Anything odd:** S_note: 51 is pi0's own suffix length (50 actions + 1 state token, T007); the loaded config's chunk is 50 (shapes.json own_S); timing_correction: CUDA-event column (s_sweep.csv median_us) times 50 back-to-back calls per event pair; for q it still carries ~15-20 us per call of launch overhead at S<=51 (event_overhead_us in s_sweep_ncu.csv). Kern; gate_note: read when this meta.json was written (after the run); the queue refused non-MAXN before it started and ran jetson_clocks

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
