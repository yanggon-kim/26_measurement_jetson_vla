# 20260929_ANALYTIC_corun_feasibility

* **What:** analytic co-run feasibility (owner request, 2026-09-28)
* **Conditions:** date 2026-09-29T06:31:35Z, power mode MAXN (every source run)
* **Command:** `python3 bench/analytic_corun.py --out results/20260929_ANALYTIC_corun_feasibility`

Headline — first rows of `bounds.csv` (6 rows):

| model | bytes | t_A_floor_ms | t_B_floor_ms | serial_floor_ms | corun_floor_ms | corun_limited_by | overlap_headroom |
|---|---|---|---|---|---|---|---|
| pi0 | analytic | 140.45403909555517 | 34.030425348055694 | 174.48446444361088 | 151.76532557910346 | compute | 1.14969914094551 |
| pi0 | measured | 140.45403909555517 | 43.81394356430363 | 184.2679826598588 | 159.22227556409024 | bandwidth | 1.1573002710018871 |
| pi05 | analytic | 162.98225544764347 | 58.36259241478636 | 221.34484786242984 | 181.0970237530797 | compute | 1.2222445365210797 |
| smolvla | analytic | 3.230985053608577 | 10.849309222808984 | 14.080294276417561 | 14.080294276417561 | bandwidth | 1.0 |
| groot_n17 | analytic | 18.009502487282354 | 69.15307912732705 | 87.1625816146094 | 85.40634665813198 | bandwidth | 1.0205632839384564 |
| groot_n17 | measured | 69.13586173787806 | 62.7844455112818 | 131.92030724915986 | 131.92030724915986 | bandwidth | 1.0 |

**Anything odd:** notes: ['Owner question (2026-09-28): "Let\'s say we have some amount of compute and some amount of bandwidth and capacity\nrequirements for action and prefix at the same time. Check whether they can be run 

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
