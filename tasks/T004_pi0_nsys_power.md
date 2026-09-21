# T004 — π0 kernel timeline, SM utilisation, roofline and power (E5, E6, E7)  · priority P1 · depends on: T003

Model: π0 (openpi PyTorch, same checkpoint and harness as T002/T003). Modes: eager and the best compiled mode of
T003. N = 4 and N = 10. MAXN + `jetson_clocks`.
T010 (E11) runs before this task: reuse its `bench/phase_loops.py` (prefix-only / action-expert-only / full loops) for
step 4, and if its nsys captures (`results/<date>_E11f_pi0_pipelining/raw/`) cover a configuration of step 1, analyse
that capture here instead of taking a new one (reference it in `meta.json.notes`; do not copy the `.nsys-rep`).
0. Physical floor per region → `floor.csv` (protocol §9), before any profiling.
1. E5 / nsys (protocol §3.1): capture window = 1–3 steady-state inferences; commit the `.nsys-rep` (and the
   `.sqlite` if under 50 MB compressed) under `raw/` per protocol §3.5 / §6; export the CSVs; compute kernels per step, kernel time vs region time, GPU idle
   fraction, gap histogram, top-20 kernels → `kernels_per_region.csv`.
2. E5 / ncu (protocol §3.2–3.3; on this device: `sudo $(which ncu)`, the clock-control value T013 selected, stall
   reasons from the `raw` page, no DRAM counters — off-chip bytes from the L2 sysmem-miss sectors): write `bench/ncu_target.py` and `bench/parse_ncu.py`
   (`docs/03_instruction_level_analysis.md` §11 items 1–2). One filtered run per region, in this order:
   `ae_layer[0]/`, `ae_step[0]/`, `vlm_prefix/`, `vision/`; eager at N = 4 first, then the best compiled mode.
   (One `ae_step` is the same work at N = 4 and N = 10: capture it once per mode.) Export both pages; produce
   `ncu/columns.txt`, `ncu/kernels.csv`, `ncu/region_utilisation.csv` with SM occupancy / activity, per-pipe
   utilisation and warp-stall reasons. Never unfiltered; if a region is too large, follow the fallback in §3.2
   and report the kernel-time coverage. Commit every `.ncu-rep` and both full page exports (protocol §3.5 / §6).
3. E6: roofline CSV per the plan for `vision`, `vlm_prefix`, `ae_step` (analytic rows, and `source = ncu` rows built
   from SM-throughput FLOPs and sysmem-miss bytes — never from `Memory Throughput`, which is L2 here), against the
   Orin's own peaks (needs `hw_constants.json`; if T005 has not run, use the spec values and mark them).
4. E7: power per rail for idle / prefix-only / action-expert-only / full loops; energy per inference and per
   denoising step. All four rails from sysfs (protocol §4), the memory rail reported as absolute and above-idle
   `VDDQ_VDD2_1V8AO (regulator input)`. Capped modes per protocol §1: only `MODE_50W`, only if it switches without
   a reboot, as the last item of the last MAXN session; `MODE_30W` / `MODE_15W` are not part of this task.
Result directories: `results/<date>_E5_pi0_<mode>_N<N>/`, `results/<date>_E6_pi0/`, `results/<date>_E7_pi0_<mode>_<powermode>/`.
Acceptance: `reports/E5_pi0_timeline.md` answering (a) kernels executing vs GPU waiting and (b) region-weighted
and kernel-weighted SM occupancy, pipe utilisation and stall reasons per region, next to `GR3D_FREQ`, set against
the paper's 22 %, ending with the root-cause chain of protocol §10; `reports/E6_pi0_roofline.md` (short; the
CSV is the deliverable); `reports/E7_pi0_power.md` with W per rail per phase and mJ per step. Every result directory has its raw files
committed under `raw/` and listed in `meta.json.raw_files` (E5: `.nsys-rep`, `.ncu-rep`, `torch.profiler` trace if
taken; E7: raw tegrastats logs; all: command stdout), no committed file over 50 MB, repository-size lines in
`STATUS.md` (protocol §6f). Self-review (§11).
