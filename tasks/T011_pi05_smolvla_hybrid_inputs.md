# T011 — π0.5 and SmolVLA: inputs for the hybrid GPU + PIM model (E11)  · priority P2 · depends on: T006 (step 1), T010 · run right after T006 (task order table in `docs/01_measurement_plan.md`)

Everything T010 did for π0, for **π0.5** (openpi PyTorch, checkpoint `pi05_libero`, S = 50, 18 expert layers,
hidden 1024, default N = 10) and **SmolVLA** (lerobot at the pinned commit, the checkpoint used in T006, S = 50,
16 expert layers — self-attention and cross-attention layers, hidden 720, default N = 10). One model completely
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
