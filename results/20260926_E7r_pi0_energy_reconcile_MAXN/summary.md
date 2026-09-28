# 20260926_E7r_pi0_energy_reconcile_MAXN

* **What:** T016 step 4 (EV-10): E7 energy reconciliation, pi0 bf16_both N=10
* **Conditions:** date 2026-09-27T03:01:19Z, power mode NV Power Mode: MAXN, rung / mode bf16_both
* **Command:** `/home/yanggon/venvs/pi0/bin/python bench/e7r_reconcile.py --outdir results/20260926_E7r_pi0_energy_reconcile_MAXN --measure --seconds 45 --rounds 2`

Headline — first rows of `E7_all_dirs_timed_window.csv` (132 rows):

| dir | phase | samples | E_committed_mJ | E_timed_window_mJ | change_pct |
|---|---|---|---|---|---|
| 20260924_E7_groot_n15_a… | prefix | 1281 | 5241.3 | 5544.0 | 5.8 |
| 20260924_E7_groot_n15_a… | ae | 1358 | 2705.9 | 3020.5 | 11.6 |
| 20260924_E7_groot_n15_a… | full | 1646 | 6635.3 | 8709.4 | 31.3 |
| 20260924_E7_groot_n15_a… | prefix | 1292 | 3957.9 | 4206.4 | 6.3 |
| 20260924_E7_groot_n15_a… | ae | 1384 | 2059.8 | 2340.2 | 13.6 |
| 20260924_E7_groot_n15_a… | full | 1549 | 5409.9 | 6684.5 | 23.6 |

**Anything odd:** nothing flagged in meta.json.

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
