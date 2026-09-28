# 20260925_E12b_pi0_dram_bytes_MAXN

* **What:** E12b follow-up: measured DRAM bytes vs E6 bytes vs EMC-implied bytes (analysis/E12b_review.md, owed question)
* **Conditions:** date 2026-09-26T06:29:34Z, power mode NV Power Mode: MAXN
* **Command:** `bash bench/run_ncu_e12b_dram.sh results/20260925_E12b_pi0_dram_bytes_MAXN`

Headline — first rows of `dram_bytes.csv` (2 rows):

| stage | workloads_per_frame | L2_read_requests_GB_per… | L2_read_hit_share | measured_read_GB_per_fr… | measured_write_GB_per_f… | measured_total_GB_per_f… | E6_analytic_GB_per_frame |
|---|---|---|---|---|---|---|---|
| prefix_alone | 1.0 | 73.9964 | 0.759 | 17.8025 | 3.833 | 21.6356 | 5.8467 |
| expert_alone | 9.0 | 14.9999 | 0.532 | 7.027 | 1.1868 | 8.2138 | 6.3797 |

**Anything odd:** caveat: graph-level profiling: one workload per graph launch (whole stage). Writes are an UPPER bound (L2 write sectors, not write-backs). Durations are profiled, NOT latencies -- the GB/s columns use the unp

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
