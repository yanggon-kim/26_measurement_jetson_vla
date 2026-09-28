# 20260925_E12_pi0_green_N10_MAXN_eager

* **What:** E12 Step 3 -- SM partitioning via CUDA green contexts
* **Conditions:** power mode NV Power Mode: MAXN, model pi0, rung / mode eager
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun_green.py --rung eager --steps 10 --sms 4,6,8,10,12,16 --seconds 20 --outdir results/20260925_E12_pi0_green_N10_MAXN_eager`

Headline — first rows of `sm_sensitivity.csv` (12 rows):

| stage | want_sm | granted_sm | rung | N | power_mode | n | median_ms |
|---|---|---|---|---|---|---|---|
| prefix | 4 | 4 | eager | 10 | NV Power Mode: MAXN | 30 | 1425.0715 |
| expert | 4 | 4 | eager | 10 | NV Power Mode: MAXN | 34 | 588.6407 |
| prefix | 6 | 6 | eager | 10 | NV Power Mode: MAXN | 30 | 993.8681 |
| expert | 6 | 6 | eager | 10 | NV Power Mode: MAXN | 35 | 584.0688 |
| prefix | 8 | 8 | eager | 10 | NV Power Mode: MAXN | 30 | 779.7903 |
| expert | 8 | 8 | eager | 10 | NV Power Mode: MAXN | 35 | 587.5578 |

**Anything odd:** rung_note: eager; not_measured: ["Two stages running CONCURRENTLY in separate partitions. This run establishes each stage's own SM demand, which is the necessary condition for contention; it does not exhibit interference. Running bo; caveat: sm_efficiency below ~0.3 for a stage means that stage cannot be the victim OR the cause of CTA-level contention on this device, because it does not use the SMs it already has. That is a statement abou

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
