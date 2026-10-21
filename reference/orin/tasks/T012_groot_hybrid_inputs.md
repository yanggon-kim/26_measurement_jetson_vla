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

status: done 2026-09-24

**Results.** Per-segment DiT block data for all three GR00T versions, via
`bench/groot_segments.py` (new) reusing `bench/host_segments.py`'s timing engine unchanged,
so the numbers are comparable with π0's rather than produced by a second harness:

| version | run | model_inputs rows | checker |
|---|---|---:|---|
| N1.7 | `results/20260924_E11a_groot_n17_hostseg` | — | **CHECK_OK** |
| N1.6 | `results/20260924_E11a_groot_n16_hostseg` | — | rejected: no `groot_n16` in `MODELS` |
| N1.5 | `results/20260924_E11a_groot_n15_hostseg` | 162 | rejected: no `groot_n15` in `MODELS` |

Report: `reports/E11_models.md`. bench commit `26322ec0c513`.

**Scope, stated rather than implied.** GR00T gets E11a only. E11c/d/e/f are built on the
LeRobot phase loops and a `transformers` KV cache, and GR00T has neither — its prefix is a
whole VLM invoked as one unit and its action head takes the backbone output, not a cache.
Porting them is separate work. What GR00T contributes is the per-segment bytes, FLOPs and
times of a DiT block, which is what the PIM model needs from that architecture.
`DiT.forward` (`dit.py:310-326`) calls `block(...)`, so `transformer_blocks[i].forward`
is a valid wrap point — checked before anything was written. Both block kinds are measured,
because odd indices are pure self-attention and even ones cross-attend onto the VLM's tokens.

**Blocked on the research side (not a defect here).** `check_model_inputs.py`'s `MODELS`
set is `{pi0, pi05, smolvla, groot_n17, none}`, so `groot_n16` and `groot_n15` rows are
rejected. The rows are written anyway, each `meta.json` records the mismatch, and the run
says so loudly. **Relabelling the rows to pass the checker would be falsifying the model
field, so it was not done.** Raised in `STATUS.md` under "Questions for the research side":
please extend `MODELS`.

**Deviations.**

1. **N1.5 needed the model's own autocast context.** `gr00t/model/policy.py:190` wraps
   n1.5's inference in `torch.autocast(cuda, bfloat16)`; n1.7 and n1.6 do not (checked: 2
   mentions against 0). Reconstructing outside that context read an fp32
   `encoder_hidden_states` into a bf16 Linear. It took three attempts to fix properly —
   first the detection, then its scope (the timing closures as well as the reconstruction).
   The capture itself was always correct: `groot_segments.py:107` calls `GP_step`, which
   mirrors `groot_phases.py`'s own autocast handling.
2. **N1.5's compiled `adalayernorm1` timing is not measured.** It is the only version
   running under `torch.inference_mode`, and inference tensors cannot be saved for backward,
   so `torch.compile` refuses them. n1.7 and n1.6 have zero such failures and n1.5's eager
   timings are unaffected. Recorded in that run's `meta.json` under `not_measured`.
3. Segment bytes verified bf16 rather than assumed: N1.7's `adaln_cond` `bytes_read` is
   9440256, which is 1536×3072 weight elements at 2 bytes plus temb 1536 at 2 bytes exactly.
   N1.5 matches after the autocast fix.

status: done 2026-09-28 — completeness audit (owner request): the 2026-09-24 scope reduction (E11a only) is withdrawn; steps 1–7 are delivered
- Step 1 complete: `results/20260928_E11a_groot_n17_hostseg` (kernel counts, nsys host sequences, sequence /
  whole_layer_graph, 8 step extras; blocks 0 / 16 cross-attention, 1 / 17 self-attention). **The DiT has 32 blocks
  (16 cross-attention), not the 16 this task assumed.**
- Step 2: `results/20260928_E11c_groot_n17_handover_xfer`. Step 3: `results/20260928_E11c_groot_n17_handover` — the head
  receives backbone features (1, 273, 2048) bf16 = 1,118,208 B, not a KV cache; the 16 cross-attention blocks re-project
  them through to_k / to_v at every denoising step (26.8 MB, 3.18 ms per step).
- Step 4: `results/20260928_E11d_groot_n17_{eager,vlm_compile}_prefixsweep` — the checkpoint accepts exactly 2 cameras
  (1 refused: wrist camera required; a 3rd is silently ignored — both in raw/refused_points.csv); prompts 14 / 17 / 75
  tokens. A first run (same-length short / typical prompts, 3-image points that were really 2-image) is superseded.
- Step 5: `results/20260928_E11e_groot_n17_{eager,vlm_compile}_mempressure`. Step 6:
  `results/20260928_E11f_groot_n17_{eager,vlm_compile}_pipelining` (compiled GPU idle not measurable inside one graph).
- Step 7: `results/20260928_E11_groot_n17_inputs` (2201 rows, 32 / 35 quantities, checker exit 0).
- Report `reports/E11_groot_n17_hybrid_inputs.md` in T010's structure; GR00T row in `reports/E11_models.md`.
- Code: `bench/groot_e11.py`, `bench/groot_segments.py` (count / nsys / time), `bench/segments_groot.py` (step extras),
  shared functions extracted from host_segments / mempressure / pipelining (LeRobot outputs verified unchanged).
- bench commit at close: 333505a.
