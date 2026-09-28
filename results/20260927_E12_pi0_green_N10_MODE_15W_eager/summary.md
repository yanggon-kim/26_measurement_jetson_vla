# 20260927_E12_pi0_green_N10_MODE_15W_eager

* **What:** E12 Step 3 -- SM partitioning via CUDA green contexts
* **Conditions:** power mode NV Power Mode: MODE_15W, model pi0, rung / mode eager
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/corun_green.py --allow-capped --rung eager --steps 10 --sms 2,4,6,8 --seconds 20 --outdir results/20260927_E12_pi0_green_N10_MODE_15W_eager`

Headline — first rows of `sm_sensitivity.csv` (8 rows):

| stage | want_sm | granted_sm | rung | N | power_mode | n | median_ms |
|---|---|---|---|---|---|---|---|
| prefix | 2 | 4 | eager | 10 | NV Power Mode: MODE_15W | 30 | 4683.2632 |
| expert | 2 | 4 | eager | 10 | NV Power Mode: MODE_15W | 30 | 1185.3097 |
| prefix | 4 | 4 | eager | 10 | NV Power Mode: MODE_15W | 30 | 4685.1211 |
| expert | 4 | 4 | eager | 10 | NV Power Mode: MODE_15W | 30 | 1149.9592 |
| prefix | 6 | 6 | eager | 10 | NV Power Mode: MODE_15W | 30 | 3306.6106 |
| expert | 6 | 6 | eager | 10 | NV Power Mode: MODE_15W | 30 | 1132.4481 |

**Anything odd:** rung_note: eager; not_measured: ["Two stages running CONCURRENTLY in separate partitions. This run establishes each stage's own SM demand, which is the necessary condition for contention; it does not exhibit interference. Running bo; caveat: sm_efficiency below ~0.3 for a stage means that stage cannot be the victim OR the cause of CTA-level contention on this device, because it does not use the SMs it already has. That is a statement abou

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
