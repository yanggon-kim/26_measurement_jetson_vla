# Note (2026-09-26) — calibration anomaly and files added after the run

* **In-session EMC calibration not reproducible today.** `calib/` (run by the harness) and two standalone
  repeats (`calib_recheck1`, `calib_recheck2`) give EMC% per hog setting that is stable (89/80/68/50/28 %),
  but the hog's own CUDA-event rate at the top varies 122.6–157.6 GB/s (E12b 2026-09-25: 178.5; this
  morning's smoke: 181.8). The EMC side is stable; the hog's 10-replay event timing is not. So the harness's
  `left_GBps` / `emc_*_GBps` columns in `sigma.csv` (which used `calib/`: 1.445 x EMC% + 15.57, max 139.8) are
  **superseded** by `sigma_derived.csv`:
  - `left_vs_E12b_stream_max_178.5` = 178.5 − the load's own byte-counted rate during the prefix;
  - EMC% during / reference (raw medians from `raw/teg_*.log`) and two conversions: E12b's fit
    (2.005 x + 0.34) and a fit from THIS session's SM-free load alone (its own byte count over 5 s vs EMC%,
    8 points): **2.321 x EMC% − 10.32, R² 0.99986**.
  The primary x-axis (the load's own byte count during the prefix) and every sigma are unaffected.
* **No-kernel proof** (`raw/load_trace_cuda_gpu_kern_sum.csv` — empty; `raw/load_trace_cuda_gpu_mem_time_sum.csv`
  — 2196 H2D + 2196 D2H memcpys): nsys `-t cuda` of the copy-engine load alone (`bench/bwload.py --kind ce --rate -1
  --seconds 3`). The CPU part of the `both` load issues no CUDA calls at all.
* Added after `meta.json` was written (so not in its `raw_files`): `calib_recheck1/`, `calib_recheck2/`,
  `sigma_derived.csv`, the two `raw/load_trace_*.csv`, this note.
