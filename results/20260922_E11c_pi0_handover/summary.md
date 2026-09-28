# 20260922_E11c_pi0_handover

* **What:** E11c — task T010
* **Conditions:** date 2026-09-22T07:25:37Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/handover.py --outdir results/20260922_E11c_pi0_handover`

Headline — first rows of `model_inputs.csv` (22 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | kv_bytes_per_layer | padded |  | 835584 | bytes |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | kv_bytes_per_layer | valid |  | 545792 | bytes |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | kv_bytes_total | padded |  | 15040512 | bytes |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | kv_bytes_total | valid |  | 9824256 | bytes |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_kv_materialise | stacked | layout=stacked | 0.570407999 | ms |
| pi0 | 45dcc8fc0e02601c8ccf055… | a3878f563ddabe86dc71d82… | t_kv_materialise | stacked | layout=stacked | 0.566232951 | ms |

**Anything odd:** notes: ['The cross-check in the task text is 18*2*P*1*256*2 B; the TOTAL row of handover.csv carries both that figure and the measured one, and they are reported, not forced.', 'kv_copied_bytes_per_step is t; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
