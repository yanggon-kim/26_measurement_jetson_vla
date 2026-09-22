# E11 — consolidated hand-off (T010)

`model_inputs.csv`: **1,589 rows, all 35 protocol §13.8 quantities present**, and
`python docs/reference_scripts/check_model_inputs.py` exits 0 (as it does for each of the eight
source runs).

Every row is restated from a part's own tidy CSV, copied under `raw/restated/<run dir>/` with
`from=<run dir>` in `variant`, per the §13.8 copy rule. Profiler reports are never copied.
`provenance.csv` lists which run contributed how many rows.

Two quantities come from outside the E11 runs: `t_preprocess` and `t_postprocess` are restated
from T002's `preprocess.csv` (`bench/measure_preprocess.py`), because `bench/regions.py` never
wraps those two regions and no E11 run produces them.

Headline corrections to the analytic model are in `reports/E11_pi0_hybrid_inputs.md`. The
largest: `T_host_layer` 0.16 ms assumed vs 1.502 ms eager / 0.177 ms as one CUDA graph;
transfers 0 + bytes/143 GB/s assumed vs 40 us + bytes/36 GB/s; synchronisation free vs 108.8 us
per stream handoff; `kv_copied_bytes_per_step` 0 assumed vs the whole 15.04 MB cache every step.
