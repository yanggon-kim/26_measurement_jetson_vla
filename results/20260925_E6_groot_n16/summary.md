# 20260925_E6_groot_n16

* **What:** E6 — task T009 step 2 (E6 for GR00T N1.5 / N1.6)
* **Conditions:** date 2026-09-25T21:39:52Z, power mode not applicable (no timing: bytes are summed from the checkpoint's tensors), model groot_n16
* **Command:** `python bench/floors_groot.py --checkpoint nvidia/GR00T-N1.6-3B --version n16 --embodiment-tag robocasa_panda_omron --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 (outdir results/20260925_E6_groot_n16)`

Headline — first rows of `floor.csv` (6 rows):

| model | region | bytes_read | flops | peak_GBps | peak_TFLOPps | floor_mem_ms | floor_compute_ms |
|---|---|---|---|---|---|---|---|
| groot_n16 | vision | 855361408 | 0 | 204.8 | 42.0 | 4.1766 |  |
| groot_n16 | vision | 855361408 | 0 | 187.47 | 33.8506 | 4.5627 |  |
| groot_n16 | vlm_backbone | 3735969664 | 0 | 204.8 | 42.0 | 18.242 |  |
| groot_n16 | vlm_backbone | 3735969664 | 0 | 187.47 | 33.8506 | 19.9284 |  |
| groot_n16 | ae_step | 2837248000 | 0 | 204.8 | 42.0 | 13.8537 |  |
| groot_n16 | ae_step | 2837248000 | 0 | 187.47 | 33.8506 | 15.1344 |  |

**Anything odd:** backfill_note: 2026-09-28 completeness audit: this directory was committed (f2531c8, 2026-09-25) without a meta.json; written from floor_inputs.json and the commit that produced it; nothing recomputed; throttled: not applicable (no timing)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
