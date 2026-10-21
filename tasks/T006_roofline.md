# T006 — Roofline per part and per kernel (E5 captures, E6, C3) — deliverable 2 · priority P0 · depends on: T005

## Why
Deliverable 2: the inputs to draw the roofline — this device's ceilings (T003), and for each model, mode and part,
**and for each kernel**, the FLOPs, the DRAM bytes and the achieved rate. On the Orin the roofline came late, only
per part, from analytic bytes that undercounted π0's prefix 3.7×, and without an `eager_fp32` row for every model.
This task fixes all three: per-kernel points from measured counters joined to nsys durations, analytic rows beside
them, both modes for every model.

## Orin counterpart
`reference/orin/reports/E6_pi0_roofline.md`, `E6_pi05_roofline.md`, `E6_smolvla_roofline.md`, `E6_groot_roofline.md`,
`E4_PARTS_ABC_GUIDE.md` §5 C3; scripts `reference/orin/bench/roofline.py`, `expert_dims.py`, `vision_flops.py`, `nsys_target.py`, `run_nsys.sh`, `parse_nsys.py`, `e13d_kernel_bytes.py` (the per-kernel byte method).

## Configuration
Maximum-performance mode, `jetson_clocks`, gate passed; `micro_peak.py` at the start of each session (its value
beside the T003 ceilings). Modes: `eager_fp32` and `best_bf16` **exactly as `results/<date>_E4i_summary/best_bf16.json`
names it**, through the same callables that are timed. Each model at its default N. The ceilings and ridge are read
from T003's `hw_constants.json` for the dtype each part runs in (fp32 / TF32 for `eager_fp32`, bf16 for `best_bf16`).

## Steps
1. **C3 — operand dimensions** (CPU-only): every expert GEMM of every model from the loaded checkpoint (M = S
   measured on a real inference, K, N, count per step), and the same for the prefix and vision GEMMs (M = tokens),
   with analytic FLOPs and unique operand bytes (weight, activation, output) per GEMM → `dims.csv`.
2. **Analytic per part**: FLOPs and bytes per part from C3 plus attention and the non-GEMM ops' shapes (record the
   formulas). Per kernel: link every kernel to its aten op and input shapes with one `torch.profiler` run
   (`record_shapes=True`; kernel ↔ op correlation), and give GEMM and attention kernels their analytic FLOPs and
   bytes; other kernels get analytic bytes where the op's shapes define them, else `n/a`.
3. **nsys capture** (protocol §3.1; this is E5's capture, which T007 analyses): per model × mode, 2 steady-state
   inferences, graph-node tracing for graphed modes; full CSV export; the per-kernel table with **every** kernel
   (launch order, name, grid, block, start, duration, NVTX region). The binaries follow T001's verdict.
4. **ncu metric-only capture for every kernel of every part** (protocol §3.2 form D, §3.3): the DRAM-byte, FLOP
   (SIMT and tensor), tensor-pipe, duration, grid / block and SM-throughput metrics T003 validated, no `--set`;
   chunked with `-s` / `-c` windows from the nsys kernel order where ncu's per-process limit or graph replay needs
   it; coverage ≥ 95 % of each part's kernel time. Clock check per run.
5. **Join and compute** → `kernel_roofline.csv` (protocol §5: ncu ↔ nsys on launch order + kernel name + grid;
   time from nsys, never ncu `Duration`), and `roofline.csv` per part with `source = analytic` and `source = ncu`
   rows (parts' time = the nsys kernel sum and, in a second row, the §2 CUDA-event region time), with the ceilings,
   ridge, % of peak compute and bandwidth, and the bound. `floor.csv` and `gap_ratio` per part (protocol §9).
6. **Plots** (matplotlib, `Agg`): one roofline per model with both modes' parts and kernels against the measured
   ceilings (log-log, bf16 and fp32 / TF32 roofs), committed as PNG next to the CSVs. The CSV is the deliverable;
   the plot is a check.

## Result directories and reports
`results/<date>_E6_dims/` (`dims.csv`), `results/<date>_E5_<model>_<mode>_nsys/` (12 directories),
`results/<date>_E6_<model>_<mode>/` (12 directories: `ncu/` raw and exports, `kernel_roofline.csv`, `roofline.csv`,
`floor.csv`, `roofline.png`, `raw/`); reports `reports/E6_<model>_roofline.md` (six; short — the table of parts,
the ten heaviest kernels, coverage, where analytic and measured bytes disagree and by how much) and
`reports/E6_roofline_all.md` (all models on one page, Thor vs the Orin's E6 values labelled Orin).

## Acceptance
- For each of the 12 (model, mode) pairs: `roofline.csv` with all three parts, both sources; `kernel_roofline.csv`
  covering ≥ 95 % of each part's kernel time (coverage stated); `floor.csv` and gap ratios; the deliverable-2 cells
  ticked in `PROGRESS.md`.
- GR00T's vision and LLM are separate parts (T004's split).
- Every measured byte and FLOP comes from a counter T003 validated, or the row says which counter and why.
- Independent review pass (P0).

## Push points
After each model's pair of E5 and E6 directories, and at the close.

## Status (appended by the on-device agent)
