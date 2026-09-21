# T004 — π0 kernel timeline, SM utilisation, roofline and power (E5, E6, E7)  · priority P1 · depends on: T003

Model: π0 (openpi PyTorch, same checkpoint and harness as T002/T003). Modes: eager and the best compiled mode of
T003. N = 4 and N = 10. MAXN + `jetson_clocks`.
0. Physical floor per region → `floor.csv` (protocol §9), before any profiling.
1. E5 / nsys (protocol §3.1): export the CSVs; compute kernels per step, kernel time vs region time, GPU idle
   fraction, gap histogram, top-20 kernels → `kernels_per_region.csv`.
2. E5 / ncu (protocol §3.2–3.3): write `bench/ncu_target.py` and `bench/parse_ncu.py`
   (`docs/03_instruction_level_analysis.md` §11 items 1–2). One filtered run per region, in this order:
   `ae_layer[0]/`, `ae_step[0]/`, `vlm_prefix/`, `vision/`; eager at N = 4 first, then the best compiled mode.
   (One `ae_step` is the same work at N = 4 and N = 10: capture it once per mode.) Export both pages; produce
   `ncu/columns.txt`, `ncu/kernels.csv`, `ncu/region_utilisation.csv` with SM occupancy / activity, per-pipe
   utilisation and warp-stall reasons. Never unfiltered; if a region is too large, follow the fallback in §3.2
   and report the kernel-time coverage.
3. E6: roofline CSV per the plan for `vision`, `vlm_prefix`, `ae_step` (analytic and ncu-based rows), against the
   Orin's own peaks (needs `hw_constants.json`; if T005 has not run, use the spec values and mark them).
4. E7: power per rail for idle / prefix-only / action-expert-only / full loops; energy per inference and per
   denoising step; repeat the full loop at MAXN, 50 W, 30 W, 15 W (last item of the session, protocol §1).
Result directories: `results/<date>_E5_pi0_<mode>_N<N>/`, `results/<date>_E6_pi0/`, `results/<date>_E7_pi0_<mode>_<powermode>/`.
Acceptance: `reports/E5_pi0_timeline.md` answering (a) kernels executing vs GPU waiting and (b) region-weighted
and kernel-weighted SM occupancy, pipe utilisation and stall reasons per region, next to `GR3D_FREQ`, set against
the paper's 22 %, ending with the root-cause chain of protocol §10; `reports/E6_pi0_roofline.md` (short; the
CSV is the deliverable); `reports/E7_pi0_power.md` with W per rail per phase and mJ per step. Self-review (§11).
