# 20260922_E11e_pi0_eager_mempressure

* **What:** E11e — task T010
* **Conditions:** date 2026-09-22T09:52:44Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/mempressure.py --outdir-prefix results/20260922_E11e_pi0`

Headline — first rows of `power.csv` (15 rows):

| phase | rail | mean_mW | n |
|---|---|---|---|
| prefix | VDD_GPU_SOC | 38765.0 | 43 |
| prefix | VDD_CPU_CV | 2388.0 | 43 |
| prefix | VIN_SYS_5V0 | 8767.6 | 43 |
| prefix | VDDQ_VDD2_1V8AO | 3312.0 | 43 |
| prefix | module | 53232.6 | 43 |
| ae | VDD_GPU_SOC | 11146.0 | 85 |

**Anything odd:** notes: ['PROXY, NOT PIM. Protocol §13.5: a software hog competes at the memory controller and in the caches and additionally takes GPU time slices or CPU cores; the two spin controls bound those side effects; backfill_note: 2026-09-27: null compile_mode replaced by its meaning; no measured value changed

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
