# T011 — π0.5 and SmolVLA: inputs for the hybrid GPU + PIM model (E11)  · priority P2 · depends on: T006 (step 1), T010 · run right after T006 (task order table in `docs/01_measurement_plan.md`)

Everything T010 did for π0, for **π0.5** (the implementation and checkpoint T006 step 1 used — LeRobot's port
by default, S = 50, 18 expert layers, hidden 1024, default N = 10; π0.5 uses AdaRMS, so the `PiGemmaRMSNorm`
conditioned branch and gated residuals T010 found unused for π0 are expected to be live here — record the branch)
and **SmolVLA** (lerobot at the pinned commit, the checkpoint used in T006, S = 50, 16 expert layers —
self-attention and cross-attention layers, hidden 720, default N = 10). Fix the two-stream concurrency harness
first (protocol §13.7 two-stream rule: one Python thread, two streams); T010's two-stream rows are not reusable. One model completely
before the other; π0.5 first. Reuse T010's `bench/` scripts (`host_segments.py`, `xfer_latency.py`,
`bw_hog.py`, `phase_loops.py`; add `--model`), the environments, fixtures and best compiled modes of T006.
Methods and schema: `docs/02_protocol.md` §13; content: `docs/01_measurement_plan.md` E11.
Per model `<m>` ∈ {`pi05`, `smolvla`}:
1. **Host segments** → `results/<date>_E11a_<m>_hostseg/`. The research side has **not** established these
   models' segment lists: derive them from the model code with the definition of protocol §13.2 (every maximal
   run of operations in an expert layer that is neither a large linear layer on the suffix tokens nor an
   attention matrix product), read every shape and dtype from the loaded model, and write `segments.csv` with
   file:line references.
   - π0.5: start from π0's six names (`rmsnorm1`, `rope`, `softmax`, `residual_rmsnorm2`, `gelu_gate_mul_up`,
     `partials_residual`) at S = 50; its norms are adaptive (conditioned on the time embedding): the dense
     layer on the conditioning vector and the scale / shift / gate operations belong to the norm and residual
     segments where the code executes them — list each operation. Its suffix embedding has no state token and
     its time MLP feeds the norms: derive `suffix_embed` accordingly.
   - SmolVLA: one segment list per layer kind (`layer_kind` = `self_attn` / `cross_attn`); name segments by
     analogy (`rmsnorm1`, `rope`, `softmax`, `residual_rmsnorm2`, `<activation>_gate_mul_up`, `residual`) and add
     what the code has beyond them. Say for the cross-attention layers where K and V come from (which VLM layer's
     cache, through which projection) and whether that projection runs once per inference or at every step.
     `partials_residual` (`mapping_only`): use 4 partials of the down projection as for π0 and mark it
     `[ASSUMPTION: 4 partials]`.
   Timing, styles, layers (layer 0 and a middle layer of each kind), sequence and `whole_layer_graph`, kernels
   per call, nsys capture, files: exactly as T010 step 1.
2. **Transfers**: do not repeat the size sweep. Measure only this model's exact boundary sizes (score matrix
   fp32 / fp16, activation tiles, `kv_bytes_total`) with the three methods → rows in
   `results/<date>_E11c_<m>_handover/xfer_exact.csv` and its `model_inputs.csv` (`xfer_latency`).
3. **Hand-over object** → `results/<date>_E11c_<m>_handover/` as T010 step 3. SmolVLA: state which VLM layers'
   K / V the expert receives and their shapes (the VLM's KV heads and head dimension, not the expert's).
4. **Prefix vs input** → `results/<date>_E11d_<m>_<mode>_prefixsweep/`, eager and best compiled, images
   {1, 2, 3} as far as the model's input format accepts × prompt {short, typical, long}; π0.5's long prompt is
   near its own (larger) maximum prompt length — record the limit you read from its config.
5. **Memory pressure and CPU** → `results/<date>_E11e_<m>_<mode>_mempressure/` as T010 step 5 (both hogs at
   25 / 50 / 75 / 100 %, both controls, CPU utilisation per phase, tegrastats).
6. **Pipelining** → `results/<date>_E11f_<m>_pipelining/` as T010 step 6, with the model's own layer count and
   default N in the host-segment replay.
7. **Consolidated** `results/<date>_E11_<m>_inputs/model_inputs.csv` incl. the restated E3 / E4 region latencies of
   T006; `python docs/reference_scripts/check_model_inputs.py results/*_E11*_<m>_*/model_inputs.csv` exits 0.
Acceptance, per model: the result directories above with the T010 file sets; `reports/E11_<m>_hybrid_inputs.md`
with the T010 report structure; and one cross-model table `reports/E11_models.md` (`model, S, P, layers,
t_host_layer eager / compiled / whole_layer_graph, t_step_extra, kernels per layer, kv_bytes_total,
t_kv_materialise, t_vision, t_vlm_prefix, prefix_throughput, e2e_throughput, prefix_slowdown at gpu_hog_100 and
cpu_hog_100`) including the π0 row from T010. Raw files under each `raw/` (protocol §3.5 / §6); repository-size
lines in `STATUS.md` after each model (§6f). Self-review (§11). A part that cannot be done is marked
"not measured" with the exact error; it does not block the rest.

status: claimed 2026-09-24

status: done 2026-09-24

**Results.** `model_inputs.csv` exists for both models and passes
`docs/reference_scripts/check_model_inputs.py` run in place:

| model | consolidated run | rows | parts_missing |
|---|---|---:|---|
| π0.5 | `results/20260924_E11_pi05_inputs` | 1159 | `[]` |
| SmolVLA | `results/20260924_E11_smolvla_inputs` | 860 | `[]` |

Parts: E11a host segments, E11c handover, E11d prefix sweep (eager + reduce-overhead),
E11e memory pressure (eager + reduce-overhead), E11f pipelining. E11b is dropped per this
task; the transfer figure each model needs comes from its own E11c.

π0's E11f was re-measured (`results/20260924_E11f_pi0_pipelining`) because both committed
runs predated the two-stream fix `773d2a9`, and `results/20260924_E11_pi0_inputs` (1564
rows) re-consolidates π0 so its table no longer cites the confounded run. Report:
`reports/E11_models.md`. bench commit `3a7f5ed64013`.

**Measured, not assumed.** P differs per model — π0 816, π0.5 968, SmolVLA 177. Reusing
π0's literal 816 would have been wrong by 5.5× on SmolVLA's prefix. S is 51 / 50 / 50.

**Deviations and defects, all disclosed.**

1. **Two quantities were WRONG in a pushed commit.** `bench/pipelining.py` called
   `nsys_target.py` without `--model`, and that flag defaults to `pi0`, so E11f's
   `gpu_idle_ae_total` and `gpu_idle_frac_ae_total` were π0's numbers under each model's
   name. Commit `493c548` carrying them reached `origin/main`. Fixed in `a49e7e9`, both
   models re-run and re-consolidated, and a correction notice is at the top of
   `STATUS.md`. Proven by exact identity: the bad rows gave kernels/step 968.25 and
   968.70, which are π0's own E5 values; π0.5's real value is 1102.70 and SmolVLA's
   1183.70. Each re-run now verifies its own identity against its own E5 capture.
2. **SmolVLA's compiled `host_sequence` segments were captured from a second eager
   instance.** Its compiled frame replays as one CUDA graph, so a Python layer capture
   records nothing. The capture is setup rather than measurement, but it means SmolVLA's
   eager-vs-compiled `host_sequence` comparison is **not** like-for-like while π0.5's is
   (π0.5's compiled frame is not graphed as a unit and captured 6/6 segments). Recorded
   per run in `meta.json`.
3. **`gpu_idle_*_ae_total` is absent under compiled modes**, because the sub-capture uses
   `--compile-mode --no-regions`. This is NOT unobtainable as previously believed: E5
   measures the same quantity per region via `--rung`
   (`results/20260924_E5_pi0_best_N4`, `gpu_idle_frac` 0.4720). Closing it is a
   consolidation change, recorded in `scratch/NOTE_e11f_compiled_idle_is_obtainable.txt`
   and not applied.
4. `model_commit` was back-filled for `20260924_E11a_pi05_hostseg` and
   `20260924_E11c_pi05_handover`, disclosed inside each `meta.json`. It was empty because
   `resolved_commit()` postdated those runs and π0.5 pins no revision.
