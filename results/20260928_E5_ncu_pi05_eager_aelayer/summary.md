# 20260928_E5_ncu_pi05_eager_aelayer

* **What:** E5 C1 ncu, action-expert layer ranges (T008 step 3) — task T008
* **Conditions:** date 2026-09-28, power mode NV Power Mode: MAXN, model pi05, rung / mode eager
* **Command:** `bash bench/run_ncu_c1.sh <model> <mode> <region> <outdir> (exact ncu command in ncu/<tag>.sh)`

Headline — first rows of `ncu/pi05_eager_ae_layer_0_details.csv` (2444 rows):

| ID | Process ID | Process Name | Host Name | thread Domain:Push/Pop_… | Id:Domain:Start/Stop_Ra… | Kernel Name | Context |
|---|---|---|---|---|---|---|---|
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |
| 0 | 109159 | python3.10 | 127.0.0.1 | 109159  "<default domai… |  | void at::vectorized_ele… | 1 |

**Anything odd:** not_measured: none; notes: ['Completeness audit 2026-09-28: T008 step 3 asked for ae_layer ncu captures (pi05 layer 0; smolvla layer 0 = self-attention, layer 1 = cross-attention), eager and best; they had never been run.', 'Fi

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
