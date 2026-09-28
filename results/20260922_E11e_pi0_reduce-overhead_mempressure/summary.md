# 20260922_E11e_pi0_reduce-overhead_mempressure

* **What:** E11e — task T010
* **Conditions:** date 2026-09-22T10:16:55Z, power mode NV Power Mode: MAXN, model lerobot/pi0_libero_finetuned_v044
* **Command:** `python bench/mempressure.py --outdir-prefix results/20260922_E11e_pi0`

Headline — first rows of `power.csv` (15 rows):

| phase | rail | mean_mW | n |
|---|---|---|---|
| prefix | VDD_GPU_SOC | 35771.6 | 61 |
| prefix | VDD_CPU_CV | 2381.5 | 61 |
| prefix | VIN_SYS_5V0 | 9587.7 | 61 |
| prefix | VDDQ_VDD2_1V8AO | 3921.1 | 61 |
| prefix | module | 51661.8 | 61 |
| ae | VDD_GPU_SOC | 10804.1 | 84 |

**Anything odd:** notes: ['PROXY, NOT PIM. Protocol §13.5: a software hog competes at the memory controller and in the caches and additionally takes GPU time slices or CPU cores; the two spin controls bound those side effects

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
