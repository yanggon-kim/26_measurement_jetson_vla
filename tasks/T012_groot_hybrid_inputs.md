# T012 — GR00T N1.7: inputs for the hybrid GPU + PIM model (E11)  · priority P3 · depends on: T009 (step 1), T010 · run after T009 (task order table in `docs/01_measurement_plan.md`)

Everything T010 did for π0, for **GR00T N1.7** (github.com/NVIDIA/Isaac-GR00T at the commit, Orin environment,
checkpoint, fixture and best compiled mode of T009; action head = 16 DiT blocks, hidden 1536, S = 41, default
N = 4). Reuse T010's `bench/` scripts with `--model groot_n17`. Methods and schema: `docs/02_protocol.md` §13;
content: `docs/01_measurement_plan.md` E11. If T009 is blocked because the model does not run, mark this task
blocked with a reference to T009 — do not substitute another model.
1. **Host segments** → `results/<date>_E11a_groot_n17_hostseg/`. The research side has **not** established this
   model's segment list: derive it from the DiT code with the definition of protocol §13.2, one list per block
   kind (`layer_kind`, e.g. self-attention block / cross-attention block, if the blocks differ), shapes and
   dtypes from the loaded model, file:line references in `segments.csv`. Expect, and verify: adaptive
   LayerNorm conditioned on the timestep embedding (its dense layer on the conditioning vector belongs to the
   norm segment), attention softmax, residual adds, the MLP activation (report whether it is gated), and per step
   the timestep encoder, the state / action encoders, the action decoder and the Euler update as extras (list
   those that are large linear layers on the S tokens separately and mark them so; they are not host segments).
   `partials_residual` (`mapping_only`, 4 partials, `[ASSUMPTION: 4 partials]`) and `attention_gpu` as for π0.
   Timing, styles, layer 0 and a middle block of each kind, sequence and `whole_layer_graph`, kernels per call,
   nsys capture, files: exactly as T010 step 1.
2. **Transfers**: only this model's exact boundary sizes, as T011 step 2 →
   `results/<date>_E11c_groot_n17_handover/xfer_exact.csv`.
3. **Hand-over object** → `results/<date>_E11c_groot_n17_handover/` (protocol §13.4). Describe what the backbone
   hands to the action head as it is in the code (we expect backbone features consumed by cross-attention rather
   than a KV cache — verify): shape, dtype, bytes, valid vs padded tokens; whether the cross-attention K / V
   projections of those features are computed once per inference or again at every denoising step, with the
   bytes and time involved; `boundary.csv` for the per-step tensors (noisy actions, state, timestep, output).
4. **Prefix vs input** → `results/<date>_E11d_groot_n17_<mode>_prefixsweep/`: camera images {1, 2, 3} as far as the
   model's input format and the checkpoint's embodiment configuration accept (say what they accept) × prompt
   {short, typical, long}; regions `vision`, `vlm_prefix` (= backbone), `ae_step` (= one DiT denoising step).
5. **Memory pressure and CPU** → `results/<date>_E11e_groot_n17_<mode>_mempressure/` as T010 step 5.
6. **Pipelining** → `results/<date>_E11f_groot_n17_pipelining/` as T010 step 6 (16 × 4 block sequences per
   inference in the host-segment replay).
7. **Consolidated** `results/<date>_E11_groot_n17_inputs/model_inputs.csv` incl. the restated E3 / E4 region
   latencies of T009; the checker exits 0 on every GR00T `model_inputs.csv`.
Acceptance: the result directories above with the T010 file sets; `reports/E11_groot_n17_hybrid_inputs.md` with the
T010 report structure; a GR00T row added to `reports/E11_models.md`. Raw files under each `raw/` (protocol
§3.5 / §6); repository-size lines in `STATUS.md` (§6f). Self-review (§11). A part that cannot be done is marked
"not measured" with the exact error; it does not block the rest.

status: claimed 2026-09-24
