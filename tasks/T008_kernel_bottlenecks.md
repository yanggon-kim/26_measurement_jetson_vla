# T008 — Per-kernel bottlenecks at the instruction level (E5b for every part, E4h, E5c) — deliverable 3b · priority P1 · depends on: T007

## Why
Deliverable 3, second half: for the dominant kernels of each part, compiled instruction-level evidence of what
they wait for — the machine code's inner loop, its instruction mix, the stall reasons mapped to instructions, and a
latency budget. On the Orin E5b covered only the action expert. Here it also covers the VLM prefix and the vision
encoder, because on Thor their speed relative to the expert decides the phase shares.

## Orin counterpart
`reference/orin/reports/E5b_summary.md`, `E5_E5b_models.md`, `E4h_vlm_prefix_kernels.md`,
`reference/orin/analysis/prefix_root_cause.md`; scripts `reference/orin/bench/pick_e5b_targets.py`,
`ncu_sass_dump.py`, `e5b_static.py`, `build_inst_inventory.py`, `sass_backward_bra.py`, `crosscheck_sass.py`,
`make_e5b_reports.py`, `analyze_vlm_prefix.py`, `e5_expert_components.py`, `micro_ae_kernels.py`.

## Configuration
As T007; SASS for this device's `gpu_arch` (T001). Procedure: `docs/03_instruction_level_analysis.md`, all sections.

## Steps
1. **Targets** (docs/03 §1) per model, mode and part from T007's tables: ≥ 60 % of each part's kernel time; a
   kernel symbol shared across models is analysed once and referenced. `sass/targets.csv` per directory.
2. **Static side** (docs/03 §2–§5): locate each binary (cuBLAS / cuBLASLt, cuDNN, `libtorch_cuda`, Triton / Inductor
   cubins from a `scratch/` cache directory), dump SASS and PTX, resource usage, backward branches, the two
   inventories (with the device's tensor-core opcodes in their own row and the FLOPs per tensor instruction),
   the annotated hot loop, pseudocode, the named algorithm, the dependency chain.
3. **Dynamic side** (docs/03 §6–§9) from T007's ncu exports (new targeted ncu runs only where a metric is missing):
   pipes incl. tensor, LSU split, stalls → instructions, the static-vs-dynamic cross-check, the latency budget (with
   T003's measured load latency if it exists).
4. **Sweeps** (docs/03 §10): the expert's S sweep (S ∈ {1, 16, own S, 256}) for the largest projection and q_proj;
   the prefix / vision shape sweep for the top GEMM of each; eager vs `best_bf16`.
5. **E4h — prefix and vision kernel classes**: per model and mode, the prefix's and the vision encoder's kernels
   grouped by class (GEMM by shape, attention, norm, elementwise, copy), time, TFLOP/s, DRAM bytes and share per
   class; the slowest shape classes against the measured ceilings.
6. **E5c — the expert's component split**: one denoising step of each model split into projections, attention,
   host segments and per-step extras, time and share, eager and `best_bf16`.
7. **Answer per part**, in one paragraph each: what do the dominant kernels of the expert, the prefix and the vision
   encoder wait for, in each mode, and how does that change with S (expert) and under compilation?

## Result directories and reports
`results/<date>_E5b_<model>_<mode>_<part>/` (docs/03 §0 file set), `results/<date>_E4h_<model>_<mode>/`
(`kernel_classes.csv`), `results/<date>_E5c_<model>_<mode>/` (`components.csv`), `results/<date>_E5b_summary/`
(`summary.csv`: `model, mode, part, kernel, share, algorithm, top_stall, top_stall_share, tensor_pipe_pct,
lsu_global_load_pct, achieved_GBps, pct_peak_bw, achieved_TFLOPps, pct_peak_compute, verdict`); reports
`reports/E5b_<model>.md` (six, docs/03 §10 structure), `reports/E5b_summary.md`, `reports/E4h_prefix_vision_kernels.md`.

## Acceptance
- Every model, mode and part has targets covering ≥ 60 % of its kernel time (or what the remainder is), each with
  the docs/03 §0 files and the verification list of docs/03 §10 satisfied.
- The per-part answers; E4h and E5c tables; the deliverable-3 "SASS" cells ticked in `PROGRESS.md`.

## Push points
After each model's E5b directories, after E4h / E5c, at the close.

## Status (appended by the on-device agent)
