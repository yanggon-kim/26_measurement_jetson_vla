# 20260926_E13c_channel_probe

* **What:** T016 step 1c (EV-02 feasibility): channel-confinement probe, read-only
* **Conditions:** date 2026-09-26T22:34:48Z, power mode NV Power Mode: MAXN

Headline — first rows of `stride_probe.csv` (15 rows):

| stride_B | lines | median_s | lines_per_s | eff_GBps_64B |
|---|---|---|---|---|
| 64 | 4194304 | 0.0019167 | 2188326442 | 140.053 |
| 128 | 4194304 | 0.0019436 | 2158043361 | 138.115 |
| 256 | 4194304 | 0.0017263 | 2429598870 | 155.494 |
| 512 | 4194304 | 0.0025711 | 1631354582 | 104.407 |
| 1024 | 4194304 | 0.0028374 | 1478225089 | 94.606 |
| 2048 | 4194304 | 0.004877 | 860015820 | 55.041 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
