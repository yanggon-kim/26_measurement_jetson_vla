# Earlier E1 passes, kept only as evidence of the bandwidth spread

These are `latency_summary.csv` files from earlier runs of `bench/micro_peak.py` in the same
session. They were produced by **earlier versions of the script** and are **not** the result of
this run — they are committed because the run-to-run bandwidth variation described in
`reports/E1_E2_microbench.md` would otherwise rest on numbers that appear in no file.

| file | what it was | peak bandwidth it produced |
|---|---|---|
| `pass1_prototype_latency_summary.csv` | first prototype: no numerical verification of the GEMMs, only `clone` + `axpy` for bandwidth | 183.4 GB/s (axpy) |
| `pass2_latency_summary.csv` | added GEMM verification and per-row clock/temperature sampling | 161.8 GB/s (mul_scalar_fp32) |
| `pass3_latency_summary.csv` | added the eight-pattern bandwidth sweep | 189.1 GB/s (axpy) |

Do not use these as results. The authoritative numbers for this run are in
`../../latency_summary.csv` and `../../hw_constants.json`. The per-process distribution taken
with the committed probe is `../bandwidth_modes.jsonl` (22 fresh processes).
