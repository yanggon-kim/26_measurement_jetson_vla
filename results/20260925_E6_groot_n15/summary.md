# 20260925_E6_groot_n15

* **What:** E6 — task T009 step 2 (E6 for GR00T N1.5 / N1.6)
* **Conditions:** date 2026-09-25T21:39:52Z, power mode not applicable (no timing: bytes are summed from the checkpoint's tensors), model groot_n15
* **Command:** `python bench/floors_groot.py --checkpoint nvidia/GR00T-N1.5-3B --version n15 --api old --embodiment-tag oxe_droid --data-config oxe_droid --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 (outdir results/20260925_E6_gro`

Headline — first rows of `floor.csv` (6 rows):

| model | region | bytes_read | flops | peak_GBps | peak_TFLOPps | floor_mem_ms | floor_compute_ms |
|---|---|---|---|---|---|---|---|
| groot_n15 | vision | 855361408 | 0 | 204.8 | 42.0 | 4.1766 |  |
| groot_n15 | vision | 855361408 | 0 | 187.47 | 33.8506 | 4.5627 |  |
| groot_n15 | vlm_backbone | 3310714752 | 0 | 204.8 | 42.0 | 16.1656 |  |
| groot_n15 | vlm_backbone | 3310714752 | 0 | 187.47 | 33.8506 | 17.66 |  |
| groot_n15 | ae_step | 2137612288 | 0 | 204.8 | 42.0 | 10.4376 |  |
| groot_n15 | ae_step | 2137612288 | 0 | 187.47 | 33.8506 | 11.4024 |  |

**Anything odd:** backfill_note: 2026-09-28 completeness audit: this directory was committed (f2531c8, 2026-09-25) without a meta.json; written from floor_inputs.json and the commit that produced it; nothing recomputed; throttled: not applicable (no timing)

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
