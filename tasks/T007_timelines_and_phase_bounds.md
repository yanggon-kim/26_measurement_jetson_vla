# T007 — Timelines and what each part is bound by (E5 analysis, E4g C1 / C2) — deliverable 3a · priority P1 · depends on: T006

## Why
Deliverable 3, first half: for each model, mode and part, how much of the time kernels execute versus the GPU
waiting for the CPU, and — while kernels execute — what they are bound by: SM or memory throughput, occupancy,
**tensor-pipe use**, and the warp-stall reasons. On the Orin this answer was conditional (the LeRobot experts were
issue-starved when eager and memory-latency-bound only after compilation; GR00T's were memory-led throughout,
`reference/orin/reports/E4_PARTS_ABC_GUIDE.md` §5), and it had four gaps this task closes: ncu ran at a different
rung from the best one, only the first 40 kernels per part were captured, tensor-pipe metrics were never kept, and
GR00T's prefix range contained vision.

## Orin counterpart
`reference/orin/reports/E5_timeline.md`, `E5_E5b_models.md`, `E4g_c1_c2_bottlenecks.md`; scripts
`reference/orin/bench/parse_nsys.py`, `analyze_step_kernels.py`, `count_step_kernels.py`, `run_ncu_c1.sh`,
`analyze_c1.py`, `c1_capture_ok.py`, `c1_validity_check.py`, `phase_utilisation.py`, `parse_ncu.py`.

## Configuration
As T006: `eager_fp32` and `best_bf16` exactly as `best_bf16.json` names them, through the timed callables; default N;
maximum-performance mode, `jetson_clocks`, gate and the ncu clock check per run.

## Steps
1. **E5 timeline analysis** from T006's nsys exports (take a new capture only if one is missing): per model × mode
   × region — kernels per region and per denoising step (graph nodes and launches both ways for graphed modes),
   summed kernel time, GPU idle = region time − union of kernel intervals (denominator stated, protocol §3.1), gap
   histogram, kernel classes (GEMM, attention, norm, elementwise, copy, other) by time, memcpy time, the launching
   CPU thread and its CPU time per region (is the CPU the bottleneck?).
2. **C1 — ncu per part, every kernel** (plan E4g; protocol §3.2 form A or C, §3.3): `--set full` plus the explicit
   stall, tensor-pipe, DRAM-byte and FLOP metrics of T003 if `full` does not carry them; per part (`vision`,
   `vlm_prefix`, `ae_step[0]`) per mode per model; chunked windows so that every kernel of the part is captured
   (≥ 95 % of its kernel time; state the coverage); never unfiltered. Gates, ported from the Orin and run on every
   capture: the capture has every requested metric and the expected kernel count (`c1_capture_ok.py`), eager and
   best captures have different kernels where the modes differ and vision and prefix captures do not overlap
   (`c1_validity_check.py`).
3. **Per-kernel and per-part tables**: `ncu/kernels.csv` (protocol §3.3, one row per launch, joined to the nsys
   launch of the same configuration) and `ncu/region_utilisation.csv` with kernel-weighted and region-weighted
   values of: SM throughput, memory throughput (named for what it measures here), achieved occupancy, tensor-pipe
   utilisation, the full stall breakdown, DRAM GB/s from validated bytes ÷ nsys duration; the per-kernel verdict
   (ncu's speed-of-light rule, and intensity vs ridge), and the share of each part's kernel time in each class
   (compute-, memory-, latency-bound; launch-bound for kernels shorter than the launch gap).
4. **C2 — eager → best per part**: which bottleneck the optimisation removed, with the evidence, one row per
   (model, part, bottleneck before, fix, bottleneck after, evidence). Counters only; times come from T005.
5. **Root-cause chain** (protocol §10) for each part of each model in both modes, and the gap table against the
   T006 floor.

## Result directories and reports
`results/<date>_E5_<model>_<mode>_timeline/` (derived tables from T006's captures: `kernels_per_region.csv`,
`gaps.csv`, `timeline_summary.csv`, `cpu_thread.csv`; it cites T006's capture directory), `results/<date>_E4g_c1_<model>_<mode>/`
(ncu raw and exports per part, `kernels.csv`, `region_utilisation.csv`), `results/<date>_E4g_c1_summary/`
(`c1_bounds.csv`: `model, mode, part, n_kernels, coverage, kernel_ms, idle_frac, sm_pct, mem_pct, occupancy_pct,
tensor_pipe_pct, dram_GBps, pct_peak_bw, top_stall, top_stall_share, verdict`; `c2_eager_vs_best.csv`); reports
`reports/E5_timeline.md` and `reports/E4g_c1_c2_bottlenecks.md` (generated from the CSVs; Orin's C1 verdicts beside
Thor's, labelled Orin).

## Acceptance
- 6 models × 2 modes × 3 parts with timeline numbers and C1 counters at ≥ 95 % kernel-time coverage (or the
  coverage reached and why), tensor-pipe columns present, captured at the exact `best_bf16`.
- The C1 gates pass on every capture (their output committed).
- A root-cause chain per part and mode; the C2 table; the deliverable-3 "ncu" cells ticked in `PROGRESS.md`.

## Push points
After each model's timeline and C1 directories, after the summary, at the close.

## Status (appended by the on-device agent)
