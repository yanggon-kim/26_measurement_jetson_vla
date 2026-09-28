# 20260925_E12b_pi0_emc_calibration_MAXN

* **What:** E12b M4 -- tegrastats EMC_FREQ utilisation calibrated against a copy hog (T015)
* **Conditions:** power mode NV Power Mode: MAXN, rung / mode n/a (device copy hog only)
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e12b_m4.py --mode calib --outdir results/20260925_E12b_pi0_emc_calibration_MAXN`

Headline — first rows of `bandwidth.csv` (12 rows):

| variant | split | condition | source | metric | median | p5 | p95 |
|---|---|---|---|---|---|---|---|
| idle | full_gpu | idle | tegrastats_emc | EMC_FREQ utilisation % | 0 | 0 | 0 |
| idle | full_gpu | idle | mem_rail | VDDQ_VDD2_1V8AO (regula… | 504.0 | 403 | 605 |
| hog_178.5GBps | full_gpu | hog_sleep_0 | tegrastats_emc | EMC_FREQ utilisation % | 89 | 69 | 89 |
| hog_178.5GBps | full_gpu | hog_sleep_0 | mem_rail | VDDQ_VDD2_1V8AO (regula… | 5435.0 | 5435 | 5435 |
| hog_161.5GBps | full_gpu | hog_sleep_200000 | tegrastats_emc | EMC_FREQ utilisation % | 80 | 79 | 80 |
| hog_161.5GBps | full_gpu | hog_sleep_200000 | mem_rail | VDDQ_VDD2_1V8AO (regula… | 5032 | 5032 | 5032 |

**Anything odd:** caveat: EMC utilisation counts every memory client (CPU, display, other engines), so a stage's calibrated GB/s is an upper bound on its own traffic.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
