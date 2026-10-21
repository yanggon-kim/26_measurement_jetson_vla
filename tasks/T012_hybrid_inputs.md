# T012 — Inputs for the hybrid GPU + PIM model, all six models (E11 a–f, `model_inputs.csv`) · priority P2 · depends on: T005, T007

## Why
The research side's system model keeps vision and the VLM prefix on the GPU and runs the expert's large matrix
products in memory. Between them the GPU still runs small host segments, synchronises, and hands over the KV cache;
in the pipelined scheme it computes the next frame's prefix meanwhile. E11 measures each of those GPU-side terms on
Thor, in the machine-readable format the model reads.

## Orin counterpart
`reference/orin/tasks/T010_pi0_hybrid_inputs.md`, `T011_pi05_smolvla_hybrid_inputs.md`, `T012_groot_hybrid_inputs.md`;
`reference/orin/reports/E11_models.md`; scripts `reference/orin/bench/host_segments.py`, `groot_segments.py`,
`xfer_latency.py`, `xfer_exact.py`, `handover.py`, `prefix_sweep.py`, `mempressure.py`, `bw_hog.py`, `pipelining.py`,
`groot_e11.py`, `consolidate_e11.py`, `e11_summaries.py`, `managed_alloc.cu`.

## Configuration
Protocol §13 is the whole method. All six models, bf16: eager and `best_bf16` (plan E11). Maximum-performance mode,
`jetson_clocks`, gate passed. Model names in `model_inputs.csv`: `pi0`, `pi05`, `smolvla`, `groot_n17`, `groot_n16`,
`groot_n15`.

## Steps
1. **E11b once** (model-independent, in the π0 pass): transfers (`pageable`, `pinned`, `zero_copy`), fits, the five
   synchronisation costs, the spin finding; rebuild `managed_alloc` for this architecture (`nvcc -arch=<gpu_arch>`).
2. Per model, in the order π0, π0.5, SmolVLA, GR00T N1.7, N1.6, N1.5:
   a. **E11a** host segments: `segments.csv` from the model code (protocol §13.2), timing in three styles and three
      timings (§13.1) at layer 0 and a middle layer, kernels per call, `t_host_layer` three ways, `t_step_extra`,
      `t_launch_per_kernel`;
   b. **E11c** the hand-over object: `handover.csv`, `t_kv_materialise`, `kv_reuse.md`, `boundary.csv` (for GR00T,
      what its backbone hands the DiT, and whether the cross-attention K / V are re-projected every step);
   c. **E11d** prefix latency vs input: cameras {1, 2, 3} as the input format allows × prompt {short, typical, long},
      eager and `best_bf16`, fits;
   d. **E11e** memory pressure and CPU: `gpu_hog` / `cpu_hog` at 25–100 % and the two controls, the slowdown curve,
      CPU utilisation per phase (§13.5–§13.6, limits repeated);
   e. **E11f** pipelining: throughputs, `gpu_idle_ae_total` from the T006 / T007 nsys trace of the same
      configuration, the pipelined host-segment run as two processes and as two streams (§13.7);
   f. the consolidated `results/<date>_E11_<model>_inputs/model_inputs.csv` (all rows of a–e plus the region
      latencies of T004 / T005 restated by §13.8's copy rule), checked with
      `python docs/reference_scripts/check_model_inputs.py` (exit 0).

## Result directories and reports
`results/<date>_E11b_xfer/`, `results/<date>_E11a_<model>_hostseg/`, `…_E11c_<model>_handover/`,
`…_E11d_<model>_<mode>_prefixsweep/`, `…_E11e_<model>_<mode>_mempressure/`, `…_E11f_<model>_pipelining/`,
`…_E11_<model>_inputs/`, each with its `model_inputs.csv`; reports `reports/E11_<model>_hybrid_inputs.md` (six) and
`reports/E11_models.md` (the cross-model table, Thor vs Orin labelled).

## Acceptance
- Every E11 directory has a `model_inputs.csv` that passes `check_model_inputs.py`; six consolidated files (or each
  missing model with its reason); "not measured" items listed with the failing command.

## Push points
After every E11 result directory, at the close.

## Status (appended by the on-device agent)
