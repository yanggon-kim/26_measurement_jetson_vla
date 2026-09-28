# 20260925_E5b_groot_n17_ssweep

* **What:** E5b S sweep (T009 step 3; docs/03 'S sweep (isolated kernels)')
* **Conditions:** power mode NV Power Mode: MAXN, model groot_n17

Headline — first rows of `s_sweep.csv` (8 rows):

| model | proj | proj_name | in | out | dtype | S | median_us |
|---|---|---|---|---|---|---|---|
| groot_n17 | largest | ff.net.0.proj | 1536 | 6144 | bfloat16 | 1 | 166.8 |
| groot_n17 | largest | ff.net.0.proj | 1536 | 6144 | bfloat16 | 16 | 166.08 |
| groot_n17 | largest | ff.net.0.proj | 1536 | 6144 | bfloat16 | 41 | 170.24 |
| groot_n17 | largest | ff.net.0.proj | 1536 | 6144 | bfloat16 | 256 | 228.416 |
| groot_n17 | q | attn1.to_q | 1536 | 1536 | bfloat16 | 1 | 93.856 |
| groot_n17 | q | attn1.to_q | 1536 | 1536 | bfloat16 | 16 | 92.832 |

**Anything odd:** S_note: 41 is T009's stated S for N1.7. The loaded config's action_horizon is 40 (shapes.json own_S_from_config); recorded, not reconciled -- the DiT may carry one extra (state) token. 40 vs 41 is inside the ; timing_correction: {'date': '2026-09-25', 'what': "CUDA-event column (s_sweep.csv median_us) is SUPERSEDED: it timed each call with its own event pair + synchronize, adding ~45-75 us on Orin. Kernel time per call is in ; raw_files_note: 2026-09-28 completeness audit: raw_files listing added (sha256 of every file in the directory)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
