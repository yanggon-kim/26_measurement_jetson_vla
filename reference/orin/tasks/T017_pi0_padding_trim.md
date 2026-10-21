# T017 — π0 prefix padding trim, decomposed: empty camera and prompt padding, separately and together (E14b)  · priority P1 · depends on: **T016 (done)** · MAXN only · order row 6d in `docs/01_measurement_plan.md`

This file is the whole specification (no separate `docs/` spec). Binding rules: `docs/02_protocol.md` §17, plus
§16 / §15 where this task reuses T016's DRAM-bytes method and the E12b compiled-prefix harness. Everything not fixed
below is your judgement: choose, record the choice in `meta.json` and the report, and move on.

## Why (the part you need of the research design)

283 of π0's 816 prefix tokens carry no information (your `reports/E3_pi0_eager.md` §1.5, §1.7a; E11c). The prefix is
`[base cam 256 | wrist cam 256 | empty cam 256 | prompt 48]`. The empty camera (`empty_cameras: 1`, an all −1 image
pushed through SigLIP) is 256 tokens, and 27 of the 48 prompt positions are padding (21 valid on the fixture). Both
are masked (`pad_mask = 0`), and positions are `cumsum(pad_mask) − 1`, so no valid token's RoPE position moves when
they are dropped. **In exact arithmetic, dropping them computes the same function.** It is not necessarily bitwise
equal: shorter GEMMs pick other kernels and reduction orders in bf16. LeRobot PR #4625 kept even a shape-*preserving*
version of the camera skip out of compiled bf16, because it flipped one paired LIBERO rollout (27/30 → 26/30).

The research side needs three things this measurement alone gives:
1. **Time.** The trimmed prefix's time decides the GPU-serial reference if the trim enters the GPU baseline. That is
   the user's open decision (OD-1 in the research side's records); nothing here is called "baseline".
2. **DRAM bytes.** The PIM design splits the LPDDR channels between the GPU prefix and the PIM expert. The split is
   sized by the prefix's **bytes**, not its time, and the bytes-per-token behaviour of the prefix is unknown (T016
   step 1d asks why it moves 3.7× its analytic bytes).
3. **Which half does what.** The camera slot (256 tokens, static per robot) and the prompt padding (27, varies with
   the prompt) differ in engineering cost. They also differ in expected effect, because GEMM M-tiles quantise: at a
   64- or 128-row tile, 816 → 789 tokens may change no tile count at all.

T016 step 3 already has a `pi0_bf16_both+nopad` row. T017 does not replace it. T017 decomposes the trim,
drops the camera **before the image encoder** (T016 did not require that), measures vision separately, and fixes an
action-difference pass criterion in advance. If T016's `nopad` code is usable, reuse it and say so.

## Fixed configuration

- Model: LeRobot π0 `lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6`, checkpoint
  `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`, batch 1, N = 10, **MAXN +
  `jetson_clocks`**.
- Harness: **the E12b compiled-prefix harness** exactly as in
  `results/20260925_E12_pi0_compiled_stages_N10_MAXN/` (bench `352f14f`, `bench/e12_capped_compiled_stages.py`,
  whole GPU): variant `bf16_both+hoist+owngraph_expert+dbuf_kv+owngraph_prefix+memo_681+prefix_first`. It measured
  prefix 246.79, expert 101.48, serial 348.73 ms (n = 60). Extend that script or write `bench/e14b_padtrim.py` on top
  of `bench/e12b_common.py` / `bench/e12b_corun.py`, and commit it.
- **Do not edit the installed LeRobot sources.** Implement each variant as a harness-level override in `bench/`
  (wrapper or monkeypatch). A config change (for example `empty_cameras = 0`) is acceptable only if you show it
  changes nothing else: normalisation stats, feature lists, the image order.

## Variants (labels are binding; every row and `meta.json` carries its label)

| label | prefix tokens (fixture) | what changes |
|---|---|---|
| `pad` (reference) | 816 | nothing: the harness as above |
| `nocam` | 560 | the empty camera is **never created**: no dummy image, no SigLIP / projector call for it, no 256 tokens in the prefix, no mask entry. The expert's K/V has 560 positions |
| `trimprompt` | 789 | the prompt is tokenised without padding to `tokenizer_max_length` (48): only the valid tokens (21 on the fixture) enter the prefix. The camera slot stays |
| `nopad` | 533 | `nocam` + `trimprompt` |

**Verify the counts on the fixture** (`bench/fixtures/observation.json`) from the tensors themselves, and state them.
If the fixture has ≠ 21 valid prompt tokens, use the true count everywhere below and say so.

**Compilation.** Keep static shapes: compile each distinct prefix length once (no `dynamic=True`, no
`mark_dynamic`, which would change the kernels). Record the number of compiles and their wall time. Timing uses the
fixture's length only. The equivalence set (Step 3) spans several prompt lengths, so `trimprompt` / `nopad` compile
once per distinct length there.

## Step 1 — equivalence gate in fp32 (run first; nothing is timed before it passes)

Check that each variant is the same function before any timing, on the shipped model (the checkpoint's own dtype,
eager, E8's `stock` configuration). Inputs: the Step 3 input set with its seeded noise. Metric: REL (defined in
Step 3), variant vs `pad`.
- **Gate: REL ≤ 1 × 10⁻⁴ for every input, for all three variants.** A larger value means the trim is implemented
  wrong (a valid token dropped, a position shifted, a mask misaligned). Fix it and re-run the gate. Report every
  failed attempt and its cause in the report; never time a variant that has not passed.

## Step 2 — time, bytes and handover per variant (compiled path)

**2a. Time.** Stages per variant, whole GPU, CUDA events on each stage's stream, device-side finite check on every
action tensor:
- `prefix` (vision + VLM prefill, the one-graph `owngraph_prefix` as in the reference);
- `vision` (all camera encodes + projector) and `vlm_prefix` (the Gemma prefill that builds the K/V), each timed
  on its own. Latitude: separate graphs for vision and prefill (label `split_vision`, and report their sum against
  the one-graph `prefix`) or event-record nodes captured inside the prefix graph. Say which you used;
- `expert` (the 10 denoising steps over the variant's K/V);
- `serial` (prefix then expert, one frame end to end, as in the reference `stages.csv`).
Bracket the variants in one session, **pad → nocam → trimprompt → nopad → pad → nopad → trimprompt → nocam → pad**,
with ≥ 60 frames per block. Report each variant's pooled median / p5–p95 / n, and each `pad` block's median. Report
the drift between `pad` blocks: if it is > 1 %, say so and add one more bracket. Your `pad` prefix should land within
~2 % of the reference's 246.79 ms; if it does not, report the gap and continue. Record clocks, EMC rate,
temperatures and `throttled` at every block boundary.

**2b. DRAM bytes (T016 convention, T016 Amendment 1 A4).** Per variant, same session as its `pad` reference:
`ncu --graph-profiling graph --cache-control none` with explicit `lts__` metrics only (the E12b set; never `--set`
/ `--section`). Reads = `lts__d_sectors_fill_device.sum + lts__d_sectors_fill_sysmem.sum`; writes ≤
`lts__t_sectors_op_write.sum`, an upper bound; 32 B per sector. ≥ 5 frames per stage (as
`bench/run_ncu_e12b_dram.sh`). Rows: `prefix` and `expert` per frame, plus `vision` and `vlm_prefix` if you split
the graphs. Use the columns of `results/20260925_E12b_pi0_dram_bytes_MAXN/dram_bytes.csv`, adding `variant`. The
`pad` prefix should reproduce E12b's 21.64 GB/frame within 3 %; if it does not, report the gap. If T016 step 1d's
per-kernel tooling works, add each variant's per-kernel bytes (`kernel_bytes.csv`, 1d's columns).

**2c. Prefix K/V handover.** Measure the bytes of the K/V the expert receives (the sum of `nbytes` over the cache
tensors, and their shapes). Expected, derived from E11c's 15 040 512 B / 816 tokens = 18 432 B per token:
`pad` 15 040 512, `nocam` 10 321 920, `trimprompt` 14 542 848, `nopad` 9 824 256 B. A mismatch is a finding; report
it with the shapes.

## Step 3 — action difference against the padded run (the pass criterion is fixed here, before any result)

**Input set, fixed before any run.** 64 observations from the HuggingFaceVLA/libero dataset through the same rows
API as `bench/e12_fixtures.py` (same image pipeline and normalisation). Take 8 tasks × 8 frames spread over their
episodes, choosing tasks so that **≥ 4 distinct valid-prompt lengths** occur, and include row offset 0 (the
fixture). Record every input in `inputs.csv`: row offset, episode, task_index, prompt, valid prompt tokens, and a
sha256 of the decoded images. Flow noise: a `torch.Generator` seeded per input (seed recorded), with the **same noise
tensor** for `pad` and every variant (its shape does not depend on the prefix).

**Metrics per input** i, on the action chunk `predict_action_chunk` returns (record its shape; if more dimensions
are returned than the environment uses, compute on both and state which is primary: the ones the environment uses):
`maxabs_i = max |a_variant − a_pad|`, `rel_i = ‖a_variant − a_pad‖₂ / ‖a_pad‖₂`, and `bitwise_i` (all elements
identical). Over the set: **MAXABS = max_i maxabs_i**, **REL = max_i rel_i** (also report medians), and the count of
bitwise-identical inputs.

**Noise floor F.** The deviation the rung already accepts: the same metrics between the shipped model (`stock`,
eager, padded) and `bf16_both` compiled `pad` on the same inputs and noise, giving `F_maxabs`, `F_rel`. E8 found that
deviation accuracy-neutral (249 vs 252 / 400).

**Controls.** (i) `pad` compiled vs `pad` compiled, a second pass in the same process: expected bitwise on all
64. If it is not, the GPU is non-deterministic here, the bitwise question for the variants is moot, and you say so.
(ii) Step 1's fp32 gate numbers, reported beside these.

**Verdict per variant (compiled `bf16_both`, variant vs `pad`), pre-registered:**
- **bitwise**: all 64 inputs bitwise identical;
- **PASS**: not bitwise, and MAXABS ≤ F_maxabs **and** REL ≤ F_rel. The trim moves the actions less than the rung
  itself does;
- **FAIL**: otherwise. A FAIL is a finding to report, never to tune away. For a failing variant, run the seeded
  paired LIBERO comparison with T016 step 2's harness (`libero_spatial`, 10 tasks × 10 episodes, the same seeds in
  both arms, `pad` vs the variant) and report successes, Wilson 95 % intervals and the exact McNemar p.

## Pre-registered expectations (research side, 2026-09-26, before any T017 measurement; not moved after results; not acceptance criteria)

Sources: E4c split of the `bf16_both` rung (vision 56.74 + `vlm_prefix` 192.86 = 249.60 ms,
`results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv`); E3 eager SigLIP per image 80.61 / 80.14 / 80.08 ms (equal
within 0.7 %); the reference harness's 246.79 / 101.48 / 348.73 ms; E12b bytes 21.64 GB; the research side's
brainstorm cases 003 §4.4 and 004 §4.1. Their arithmetic is `[BRAINSTORM]`, not a measurement: token-linear scaling
bounds the trimmed prefix at **~164 ms** (vision cut by a third) to **~183 ms** (vision held whole), i.e. ratios
0.656–0.732 of the padded prefix. ρ below is always **variant / `pad` in the same session**.

| # | quantity | bands and what each means |
|---|---|---|
| 1 | ρ_prefix(`nopad`) | < 0.64: faster than token-linear, so recheck the trim · **0.64–0.74: inside the ~164–183 ms bracket** · 0.74–0.90: saves less than the bracket; fixed costs (weight streaming, launch) dominate · > 0.90: little time saved |
| 2 | ρ_vision(`nocam`) (`trimprompt` should leave vision at ~1.00) | 0.62–0.72: the per-image cost is equal in compiled mode too (2/3, as in eager E3) · > 0.72: a fixed per-call cost, or the dummy image was cheaper · < 0.62: unexpected, recheck |
| 3 | ρ_prefix(`trimprompt`) | ≥ 0.99: tile-quantised; 816 → 789 changes no M-tile count at 64 or 128 rows, so the prompt trim alone buys ~nothing · 0.95–0.99: about token-linear (0.974) · < 0.95: unexpected |
| 4 | ρ_prefix(`nocam`) − ρ_prefix(`nopad`) (what the prompt trim adds once the camera is gone) | ≤ 0.01: tile-quantised (560 and 533 give the same tile count at 64 and 128 rows) · 0.01–0.04: about token-linear (0.026) · > 0.04: unexpected. Also report the interaction: savings(`nopad`) − savings(`nocam`) − savings(`trimprompt`) |
| 5 | bytes ratio, prefix, `nopad` / `pad` | ≤ 0.72: bytes shrink at least with the M-tile count (per-token traffic, including re-reads, dominates) · **0.72–0.80: as case 003's padding row assumes (16.05 / 21.64 = 0.742)** · > 0.80: a large token-independent part (weights read once); case 003's padding row is optimistic on bytes |
| 6 | ρ_expert(`nopad`) | ≥ 0.97: expected; the K/V is 15.0 of 638 MB the expert reads per step (E3 floor), ≤ 2.4 % · < 0.97: attention over the prefix keys costs more than its bytes |
| 7 | serial frame, `nopad` | `[BRAINSTORM]` ≈ 260–282 ms (rows 1 and 6 applied to 246.79 + 101.48); the reference is 348.73 ms |
| 8 | K/V handover | exactly the Step 2c bytes |
| 9 | action verdicts (Step 3) | fp32 gate passes for all three; compiled bf16 **not bitwise** for any of the three (each changes the Gemma GEMMs' M, and so possibly their kernels) but **PASS** for all three. If the two real cameras are encoded one call each, their vision embeddings should stay bitwise equal under `nocam`; report whether they do |

The research side's prior, stated so that it can be wrong: rows 1 and 5 land in their middle bands, and row 3 in its
first. Your job is the measurement, not the prior.

## Non-negotiables

1. **Honesty.** Do not tune implementations, input sets, seeds or tolerances toward a band or a verdict. The input
   set and seeds are fixed before the first compiled run. Every run is reported, including failed gates and FAILs.
2. **No silent NaN.** Every action tensor of every timed, checked and ncu run is finite-checked, without a host
   sync inside a timed loop. A run with any non-finite value is invalid and reported, never dropped.
3. **Gate before timing.** No variant is timed or profiled before it passes Step 1.
4. **Complete provenance.** `meta.json` has no null: `bench_commit`, model and checkpoint commits, rung, the full
   variant label (harness tokens + `pad` / `nocam` / `trimprompt` / `nopad`), prefix tokens, K/V bytes, compile
   count, seeds, `power_mode`, `sm_count`, clocks, EMC rate, `throttled`.
5. **Variants, never baseline; MAXN only.** No row is called "baseline" or "optimised baseline" (whether the trim
   enters the GPU baseline is the user's decision). No reboot, no boot-configuration change.

## Outputs

- `results/<date>_E14b_pi0_padtrim_stages_MAXN/`: `stages.csv` (variant, stage, n, median, p5, p95, mean;
  stages `vision`, `vlm_prefix`, `prefix`, `expert`, `serial`), `blocks.csv` (every bracket block with its state
  record), `kv_handover.csv` (variant, prefix tokens, measured bytes, expected bytes, shapes), `raw/` latency
  samples, `meta.json`.
- `results/<date>_E14b_pi0_padtrim_bytes_MAXN/`: `dram_bytes.csv` (E12b columns + `variant`), `ncu/` raw exports,
  `kernel_bytes.csv` if made, `meta.json`.
- `results/<date>_E14b_pi0_padtrim_equiv/`: `inputs.csv`, `equiv.csv` (one row per input × variant × comparison
  {fp32 gate, bf16 variant vs pad, floor stock vs bf16, pad repeat}: maxabs, rel, bitwise), `verdicts.csv`,
  `raw/actions_<variant>_<precision>.npz` (the action chunks; never `.pt`, which `.gitignore` excludes), `meta.json`.
  If a FAIL triggered LIBERO: `…_E8s_pi0_libero_spatial_<variant>_seeded/` as in T016 step 2.
- Report: `reports/E14b_pi0_padtrim.md`. Lead with the pre-registered table (rows 1–9) answered by number, then
  per variant: prefix / vision / `vlm_prefix` / expert / serial times, bytes, K/V bytes, the verdict. Then the
  comparison with T016 step 3's `nopad` row (same implementation or not, and the numbers side by side), then the
  conditions and anomalies.

## Acceptance

- The fp32 gate passed for all three variants, with every failed attempt reported.
- All four labels, including `pad`, have times for every stage from one bracketed session (n ≥ 120 per variant
  pooled, `pad` ≥ 180), with drift reported. They have DRAM bytes (prefix and expert, reads and writes) from the
  same session and K/V handover bytes checked against 2c.
- Step 3: 64 inputs spanning ≥ 4 prompt lengths (or the most the dataset offers, stated). MAXABS, REL and the
  bitwise count for every variant; the floor F; both controls; a verdict per variant; LIBERO for any FAIL.
- Rows 1–9 answered by number. Mark done only when every item holds, or the status line names the item that does
  not and why. Push the results when they land.

**When:** strictly after T016 is done (or has a status line closing it, even as `blocked`), and never interleaved
with a T016 step. T017 is MAXN work: if the device is in a capped mode when T016 finishes, finish those queues first
and take T017 once MAXN is restored. Never ask for a reboot for it. If the human owner gives you a different order
directly, the owner's order wins; say in `STATUS.md` where T017 sits.

status: claimed 2026-09-27
status: done 2026-09-27
- Report `reports/E14b_pi0_padtrim.md` (rows 1–9 first). Results `results/20260927_E14b_pi0_padtrim_equiv` (inputs.csv,
  gate.csv, equiv.csv, verdicts.csv, raw/actions_<variant>_{stock,fp32,bf16}.npz), `…_E14b_pi0_padtrim_stages_MAXN`
  (stages.csv, blocks.csv, kv_handover.csv, vision_bitwise.csv, raw/latency_samples.csv), `…_E14b_pi0_padtrim_bytes_MAXN`
  (dram_bytes.csv, ncu/). Input set `bench/fixtures/e14b` (64 obs, 8 tasks × 8 frames, 8 prompt lengths 6–21), fixed
  and committed before the first run. Code `bench/e14b_padtrim.py`, `bench/e14b_inputs.py`, `bench/run_ncu_e14b.sh`,
  variants from `bench/pi0_padtrim.py` (shared with T016 step 3's `nopad`; LeRobot untouched).
- Rows: 1 ρ_prefix(nopad) **0.734** (middle band); 2 ρ_vision(nocam) **0.665**, trimprompt 0.999; 3 ρ_prefix(trimprompt)
  **1.028** (≥ 0.99 band, a slowdown); 4 **−0.010**, interaction +0.018; 5 prefix bytes **0.792** (0.72–0.80); 6
  ρ_expert(nopad) **0.653** (< 0.97); 7 serial nopad **248.5 ms** (pad 347.6); 8 K/V bytes **exact** for all four;
  9 fp32 gate PASS all three (worst REL 1.5e-6), compiled bf16 0/64 bitwise and **PASS** all three (F: MAXABS 0.296,
  REL 0.0245), pad repeat bitwise 64/64, real-camera embeddings bitwise under nocam 10/10. No FAIL, so no LIBERO run.
- Deviations / failed attempts: the first gate ran the shipped model as loaded (3.61 B params bf16) and failed at REL
  ~2e-3 on the 4-input smoke (bf16 arithmetic); re-run with the model cast to fp32 as the step's heading requires, the
  checkpoint-dtype pass kept as a reported attempt and as the floor's stock side. A duplicate queue launch (two GPU jobs
  for < 1 min, both killed, smoke outputs discarded) preceded the valid run. kernel_bytes.csv (optional) not made.
- bench commit at close: fa805b0. Repository: committed results 534M; size-pack 158.10 MiB + loose 88.64 MiB.
