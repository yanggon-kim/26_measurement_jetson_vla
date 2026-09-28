# 20260926_E12_pi0_green_N10_MODE_30W_eager

* **What:** E12 Step 3 -- SM partitioning via CUDA green contexts
* **Conditions:** power mode NV Power Mode: MODE_30W, model pi0, rung / mode eager
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun_green.py --allow-capped --rung eager --steps 10 --sms 2,4,6,8 --seconds 20 --outdir results/20260926_E12_pi0_green_N10_MODE_30W_eager`

Headline — first rows of `sm_sensitivity.csv` (8 rows):

| stage | want_sm | granted_sm | rung | N | power_mode | n | median_ms |
|---|---|---|---|---|---|---|---|
| prefix | 2 | 4 | eager | 10 | NV Power Mode: MODE_30W | 30 | 3091.6629 |
| expert | 2 | 4 | eager | 10 | NV Power Mode: MODE_30W | 30 | 746.8518 |
| prefix | 4 | 4 | eager | 10 | NV Power Mode: MODE_30W | 30 | 3091.6141 |
| expert | 4 | 4 | eager | 10 | NV Power Mode: MODE_30W | 30 | 747.7113 |
| prefix | 6 | 6 | eager | 10 | NV Power Mode: MODE_30W | 30 | 2217.2139 |
| expert | 6 | 6 | eager | 10 | NV Power Mode: MODE_30W | 30 | 717.6589 |

**Anything odd:** rung_note: eager; not_measured: ["Two stages running CONCURRENTLY in separate partitions. This run establishes each stage's own SM demand, which is the necessary condition for contention; it does not exhibit interference. Running bo; caveat: sm_efficiency below ~0.3 for a stage means that stage cannot be the victim OR the cause of CTA-level contention on this device, because it does not use the SMs it already has. That is a statement abou

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
