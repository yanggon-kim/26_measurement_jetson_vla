# T010 — π0: inputs for the hybrid GPU + PIM model (E11)  · priority P1 · depends on: T003 · run right after T003 (task order table in `docs/01_measurement_plan.md`)

Goal: measured values for every GPU-side term of the research side's hybrid model (vision + VLM prefix on the
GPU, action-expert matrix products in memory), delivered as `model_inputs.csv`. Read first: `docs/00_context.md`
section "Inputs of the hybrid model", `docs/01_measurement_plan.md` E11, `docs/02_protocol.md` §13 (binding methods and
schema). Model: π0 (openpi PyTorch, the checkpoint, environment, fixture and harness of T002 / T003). MAXN +
`jetson_clocks`, batch 1, bf16. "Best compiled mode" = the one T003 reported. Do the parts in this order; commit
one result directory per commit. None of this needs `ncu`.

1. **Host segments (E11 part 1)** → `results/<date>_E11a_pi0_hostseg/`. Write `bench/host_segments.py`.
   a. Capture the real inputs of every segment at layer 0 and layer 9 during one denoising step of the fixture
      (hooks / wrappers; tensors to `scratch/`, not committed). Print P, S, T = P + S.
   b. `segments.csv` (protocol §13.2) with the π0 segments of the plan's table — `rmsnorm1`, `rope`, `softmax`
      (model dtypes, and `variant dtype=fp32_in_fp16_out`), `residual_rmsnorm2`, `gelu_gate_mul_up`,
      `partials_residual` (`mapping_only`: `partials.sum(0) + x`, partials `[4, S, 1024]` bf16 random values are
      acceptable here because the model has no such tensor), `residual`, `attention_gpu`, `cast_scores_fp32_fp16` —
      and the extras `suffix_embed`, `final_norm_out_proj`, `euler_update`. Shapes expected: S = 51, D = 1024,
      q `[1, S, 8, 256]`, k `[1, S, 1, 256]`, scores `[1, 8, S, T]`, gate / up `[1, S, 4096]`. If the loaded model
      differs (layout, dtype, an extra operation such as a mask add or a query scale inside a segment), the model
      wins: record it in `segments.csv` `notes` and in the report.
   c. Time each segment and extra: styles `eager`, `compiled`, `cudagraph` (per-segment graphs); timings
      `wall_isolated` with `gap_ms` ∈ {0, 2}, `gpu_event`, `wall_launch`; ≥ 20 warm-ups, ≥ 1,000 timed calls each;
      both layers. Then the per-layer sequence of the six mapping segments (`rmsnorm1` → `rope` → `softmax` →
      `residual_rmsnorm2` → `gelu_gate_mul_up` → `partials_residual`): `sequence` (back to back, one final
      synchronise) in `eager` and `compiled`, and `whole_layer_graph` (one CUDA-graph replay).
   d. Kernels per call from `torch.profiler` (10 calls per segment and style); one nsys capture of ten per-layer
      sequences per style (protocol §3.1, §13.1).
   Files: `segments.csv`, `segment_summary.csv` (`segment, layer, exec_style, timing, variant, median_ms, p5_ms,
   p95_ms, min_ms, n, n_kernels, bytes, flops, achieved_GBps, achieved_TFLOPps`), `host_layer_summary.csv`
   (`exec_style, key, variant, median_ms, p5_ms, p95_ms, n`), `model_inputs.csv`, `meta.json`, `summary.md`,
   `raw/segment_samples.csv.zst`, `raw/*.nsys-rep`, `raw/torch_profiler_*`, `raw/*.stdout.log`.
2. **Transfers and synchronisation (E11 part 2)** → `results/<date>_E11b_xfer/` (protocol §13.3). Write
   `bench/xfer_latency.py`. Sizes {4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 2 MB, 4 MB, 16 MB} plus the exact π0 sizes:
   the score matrix 8·S·T·4 B and 8·S·T·2 B, one activation tile S·1024·2 B, one FFN tile S·4096·2 B, and
   `kv_bytes_total` of step 3. Methods `pageable`, `pinned`, `zero_copy`; both directions; the five synchronisation
   costs; the spin-wait check. Files: `xfer_summary.csv`, `xfer_fit.csv`, `sync_summary.csv`, `model_inputs.csv`
   (`model = none`), `raw/xfer_samples.csv.zst`, and for `zero_copy` the source of whatever extension / ctypes
   code you wrote (in `bench/`) with its build command in `env/`.
3. **Hand-over object (E11 part 3)** → `results/<date>_E11c_pi0_handover/` (protocol §13.4): `handover.csv`,
   `boundary.csv`, `kv_reuse.md`, `t_kv_materialise` for `stacked` and `v_transposed`, `model_inputs.csv`.
   Cross-check to report, not to force: 18 × 2 × P × 1 × 256 × 2 B (≈ 15.0 MB at P = 816).
4. **Prefix vs input (E11 part 4)** → `results/<date>_E11d_pi0_eager_prefixsweep/` and
   `results/<date>_E11d_pi0_<best compiled mode>_prefixsweep/`. Images {1, 2, 3} × prompt {short, typical, long}
   = 9 points per mode; take extra camera images from the same LIBERO episode / frame (record which); if the
   openpi input format always encodes three image slots and pads the prompt to a fixed length, say so, report
   processed vs valid P, and still run the 9 points. Default N (10). Files: `prefix_sweep.csv`, `prefix_fit.csv`,
   `latency_samples.csv`, `model_inputs.csv`, compile logs under `raw/` for the compiled mode.
5. **Memory pressure and CPU (E11 part 5)** → `results/<date>_E11e_pi0_<mode>_mempressure/`, eager and best
   compiled (protocol §13.5–13.6). Write `bench/bw_hog.py` and `bench/phase_loops.py` (prefix-only,
   action-expert-only and full-inference loops; T004's E7 reuses them). Loads: none, `gpu_hog` and `cpu_hog` at
   25 / 50 / 75 / 100 %, `gpu_spin_control`, `cpu_spin_control`; each foreground run ≥ 60 s with tegrastats. If
   `hw_constants.json` has no measured peak bandwidth yet (T005 not done), express the background rate against
   the 204.8 GB/s spec and mark it. CPU utilisation per phase in a separate run. Files: `mempressure.csv`,
   `cpu_util.csv`, `power.csv`, `model_inputs.csv`, `raw/bw_hog_*.log`, `raw/tegrastats_*.log`,
   `raw/cpu_util_samples.csv`, `raw/latency_samples_*.csv.zst`.
6. **Pipelining (E11 part 6)** → `results/<date>_E11f_pi0_pipelining/` (protocol §13.7): (a) the three
   throughputs, eager and best compiled; (b) `gpu_idle_ae_total` from an nsys capture of 1–3 inferences per
   mode at N = 10 and N = 4 (protocol §3.1 — T004 step 1 may reuse these captures); (c) prefix-only loop
   concurrent with the host-segment sequence (18 × 10 layers per inference, 2 ms pause between segments,
   `cudagraph` per-segment style and `eager`), as two processes and as two streams in one process.
   Files: `throughput.csv`, `gpu_idle.csv`, `concurrency.csv` (`sharing, activity, condition, median_ms, p95_ms,
   max_ms, n, slowdown`), `model_inputs.csv`, `raw/`.
7. **Consolidated hand-off (E11 part 7)** → `results/<date>_E11_pi0_inputs/model_inputs.csv`: all rows of steps
   1–6 plus the E3 / E4 region latencies (N = 4 and N = 10, eager and best compiled, instrumented and slope)
   restated by the copy rule of protocol §13.8. Run
   `python docs/reference_scripts/check_model_inputs.py results/*_E11*/model_inputs.csv`.

Acceptance: the seven result directories with the files named above, each with `meta.json` (protocol §5),
`summary.md`, `raw/` listed in `raw_files`, and a `model_inputs.csv` for which the checker exits 0; every
`quantity` of protocol §13.8 is present in the consolidated file or listed as "not measured" with the failing
command; `reports/E11_pi0_hybrid_inputs.md` with (1) the quantity table, (2) measured vs the analytic values of
`docs/00_context.md` "Inputs of the hybrid model" (`T_host_layer` ≈ 0.16 ms, 20 µs per launch, 16.8 TFLOP/s and
143 GB/s effective, zero fixed transfer and synchronisation cost, slowdown 1.0) — reported as found, whichever
way it falls, (3) the limits of the memory-pressure proxy (protocol §13.5), (4) deviations. No committed file
over 50 MB; repository-size lines in `STATUS.md` (protocol §6f). Self-review (protocol §11).
If a part cannot be done (for example no working `zero_copy` route, or `torch.compile` fails on a segment), finish
the others, mark that part "not measured" with the exact error, and do not block the task on it.

status: claimed 2026-09-22

---

status: done 2026-09-22

**Result directories** (nine; all pass `docs/reference_scripts/check_model_inputs.py` run in place):
`results/20260922b_E11a_pi0_hostseg`, `…b_E11b_xfer`, `…b_E11c_pi0_handover`,
`…b_E11d_pi0_{eager,reduce-overhead}_prefixsweep`, `…d_E11e_pi0_{eager,reduce-overhead}_mempressure`,
`…b_E11f_pi0_pipelining`, `…b_E11_pi0_inputs` (consolidated: 1,564 rows, 33 of 35 quantities).

**Report**: `reports/E11_pi0_hybrid_inputs.md`. **`bench/` commit used**: `a8ae8b8` for E11a–E11d and
E11f, `eb43ed4` for E11e (the E11e re-run needed two further hog fixes). Every `model_inputs.csv`
carries the commit its own run used, and `bench/` was committed before each run so the column names
code that actually ran.

**The four quantities the hybrid model assumes and the measurement contradicts**

| assumption | measured |
|---|---|
| one kernel per host segment, 20 µs | 1–10.1 kernels; ~35 launches per layer for the six mapping segments |
| `T_host_layer` ≈ 0.16 ms | 0.1827 ms as one CUDA graph per layer; 0.4455 ms compiled sequence; 1.5290 ms eager |
| transfers free of fixed cost, 143 GB/s | `a` = 38–40 µs pinned; slope = 36 GB/s; synchronisation 0.8–68.7 µs |
| `kv_copied_bytes_per_step` = 0, `t_kv_materialise` = 0 | 15,040,512 B copied **every step** by `copy.deepcopy` at `modeling_pi0.py:918`, 5.5824 ms |

**Not measured**, each with its exact error in `meta.json.not_measured` and the report:
`t_preprocess`, `t_postprocess` (no region instrumented); `suffix_embed` under `cudagraph`
(`cudaErrorStreamCaptureInvalidated`); `final_norm_out_proj` under `compile` (inductor: "Offset
increment outside graph capture encountered unexpectedly"); GPU-idle fraction under
`reduce-overhead` (region context managers graph-break the model, so the compiled captures run
`--no-regions` and have no `ae_step` ranges to project onto — the wrinkle `tasks/T004` anticipates).

**Deviations from the task text.** None in scope or method. Three of the seven parts were measured
more than once; every superseded directory is kept with a `SUPERSEDED.md` naming its defect, per §5.

**What the three re-measurements cost, and why they are recorded rather than quietly fixed.**
An independent adversarial review of the first pass raised 40 findings. Two invalidated published
numbers, and both were invisible in the output — no error, no warning, a plausible figure:

1. The timed sequence returned `None`, so `torch.compile` deleted its body. Sized deliberately on
   this device: a compiled 2048³ bf16 matmul costs 0.0662 ms discarded and 0.6605 ms returned. The
   published "0.085 ms", the "18× win" and the "0.53× of the assumption" measured nothing. Correct
   value 0.4455 ms, i.e. **2.78× the assumption, not 0.53×** — the conclusion reverses.
2. The background hog's duty cycle never applied, so half the E11e table was one condition repeated
   four times. Fixing it exposed two further defects of my own: a closure rebinding that killed every
   `cpu_spin_control` worker, and `start_hog` discarding the hog's stderr so six tracebacks went
   nowhere and the run finished apparently fine 47 minutes later.

Three defects in a row were visible only in the output. `bench/bw_hog.py --self-test` now runs every
hog kind for two seconds and fails unless each emits samples carrying an achieved duty;
`bench/mempressure.py` calls it before loading the model. `bench/check_report_numbers.py --task=T010`
re-derives 96 figures from the CSVs and fails if any is absent from the report.
