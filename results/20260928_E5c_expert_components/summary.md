# 20260928_E5c_expert_components

* **What:** E5c (2026-09-28): action-expert time by model component, from the committed E5 nsys runs -- derived, no new measurement
* **Conditions:** date 2026-09-28T20:00:00Z
* **Command:** `python3 bench/e5_expert_components.py <E5 run>/raw/<run>.sqlite 'ae_step[' | 'ae_total' <out.csv> <label>`

Headline — first rows of `groot_n15_best.csv` (2 rows):

| busy_ms | idle_ms | kernels | label | range | span_ms | attention_ms | attention_n |
|---|---|---|---|---|---|---|---|
| 59.5054 | 0.0411 | 756 | groot_n15_best | ae_total | 59.5465 | 3.3673 | 68 |
| 60.4651 | 0.041 | 756 | groot_n15_best | ae_total | 60.5062 | 3.3702 | 68 |

**Anything odd:** inputs_note: the nsys .sqlite exports stay local (never committed, protocol §3.5); their parent run directories and meta are committed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
