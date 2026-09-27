# E14 — prefix software savings, measured (T016 step 3, row 7)

Three changes that remove prefix work the output never uses, each a labelled **variant** — whether any of them
enters the GPU reference is the user's open decision; nothing here is a baseline. Jetson AGX Orin 64 GB, MAXN +
`jetson_clocks` (GPU 1300.5 MHz, EMC 3199 MHz, 16 SMs), 2026-09-26/27.

## Row 7, answered

| # | quantity | measured | band |
|---|---|---|---|
| 7 | step 3 exact variants (`keep1`, `kvonce`) bitwise? | **bitwise: 50 / 50 inputs** for `keep1`, `kvonce` and `keep1+kvonce`, compiled (`vlm_compile`) and eager; the reference repeated in a new process is bitwise too (controls) | **bitwise, as expected** |

π0 `nopad` is not an exact-arithmetic-preserving *kernel* change (shorter GEMMs), so it was not expected bitwise: 0 / 50
bitwise, REL ≤ 1.7 × 10⁻³; closed-loop success unchanged (§1).

## 1. π0 `pi0_bf16_both+nopad` (`results/20260926_E14_pi0_nopad_MAXN`)

**What it removes.** On the fixture the prefix is `[base cam 256 | wrist cam 256 | empty cam 256 | prompt 48]` =
816 tokens, of which **283 are padding** (the empty camera's 256, all masked, and 27 of 48 prompt positions; 21
valid) — counted from the tensors (`meta.json` `tokens`). `nopad` removes the empty-camera slot from the image list
before the image encoder and slices the prompt to its valid tokens → **533 tokens**; the expert gets the trimmed
K/V. Implementation `bench/pi0_padtrim.py`, harness-level (LeRobot untouched; shared with T017). Exact in real
arithmetic: positions are `cumsum(pad_mask) − 1` and masked keys are excluded, so no valid token's position, mask
or attention set changes.

**Compilation.** `torch._dynamo` `automatic_dynamic_shapes = False`: each prefix length compiles its own static
graph, as the rung does (pad 14.1 s, nopad 15.3 s of first-call wall time; 7 unique graphs).

### Time (rung path: `prefix_once` and `model.sample_actions` on the trimmed inputs; bracketed pad, nopad, pad, nopad, pad; 30 frames per block)

| stage | `pad` median [p5–p95] (n) | `nopad` median [p5–p95] (n) | ratio |
|---|---|---|---:|
| prefix (vision + VLM prefill) | **253.0** [250.7–253.7] (90) | **189.6** [188.5–190.0] (60) | **0.749** |
| full frame (prefix + 10 denoising steps) | **420.5** [416.7–422.0] (90) | **319.5** [317.1–326.2] (60) | **0.760** |

`pad` blocks: prefix 253.4 / 253.0 / 251.0 ms (drift 0.9 %). GPU clock pinned throughout, peak 52.9 °C, 0
non-finite. The full frame saves **101 ms**: 63 ms in the prefix and 38 ms in the expert (its attention reads a
533- instead of 816-position K/V at every one of the 10 steps).

### DRAM bytes per frame (E12b graph-level ncu method, `…/bytes`; 5 frames per stage, `graphs_both` harness)

| stage | `pad` reads / writes ≤ (GB) | `nopad` reads / writes ≤ (GB) | reads ratio |
|---|---|---|---:|
| prefix | **17.65** / 3.86 | **14.64** / 2.46 | **0.830** |
| expert (10 steps) | **7.03** / 1.26 | **6.91** / 0.99 | 0.983 |
| frame total (reads + writes) | 29.81 | 25.00 | 0.839 |

The `pad` prefix reproduces E12b (17.80 / 3.83 GB) within 0.8 % / 0.9 %. Profiled kernel time (ncu, serialised):
prefix 252.2 → 182.6 ms, expert 105.3 → 69.4 ms. The prefix loses 35 % of its tokens but only 17 % of its DRAM reads —
a large token-independent part (weights streamed once per frame) remains.

### Equivalence

**Actions (≥ 50 inputs; `…/equiv`):** 5 fixtures × a seeded ±2/255 image perturbation, per-input noise seed:
**0 / 50 bitwise**, MAXABS **0.0311**, REL max **1.72 × 10⁻³** (median 1.03 × 10⁻³), 0 non-finite. Not bitwise, as
expected (shorter GEMMs pick other kernels in bf16).

**Seeded paired LIBERO (`results/20260926_E8s_pi0_libero_spatial_N10_seeded_nopad`, E8s harness):**
`libero_spatial` N = 10, same seeds as the `pad` arm: **70 / 100** [60.4, 78.1] vs 71 / 100; only-nopad 3 /
only-pad 4, exact McNemar **p = 1.00**, paired difference −1.0 pp [−6.2, +4.2]; episode agreement **0.93** (the
`pad` arm against its own repeat: 0.95). Closed-loop success is unchanged within the resolution of 100 episodes.

**Finding — that LIBERO arm was `trimprompt`, not `nopad`, and E8 / E8s fed the empty camera as a real image.**
The arm's forward was 423.3 ms against 427.5 ms (pad), and task 0 alone (one prompt length) was not faster
(424.6 ms). First hypothesis, dynamic-shape compilation (`libero_eval.py` left `automatic_dynamic_shapes` on):
**refuted** — `results/20260927_E14_pi0_nopad_libero_speed_diag` (2 tasks × 3 episodes each) gives nopad default
430.9, nopad static 426.5, pad static 426.5 ms, with 8 / 8 / 5 unique graphs (no recompile storm). Cause found in
the harness: `bench/libero_eval.py build_obs` supplies the checkpoint's `observation.images.empty_camera_0` as an
**all-zero image**, so LeRobot's `_preprocess_images` treats it as a *present* camera (**mask 1, its 256 tokens
attended**); left out, LeRobot synthesises it as in training (−1 image, mask 0) — which is what
`pi0_common.build_batch`, every timing run and the equivalence set above use. `nocam` drops only masked slots, so in
that LIBERO arm it dropped nothing and the arm measured `trimprompt` (27 tokens, ~4 ms). Consequences:
(i) that arm is reported as `trimprompt` (70 vs 71 / 100, p = 1.00, agreement 0.93);
(ii) **every E8 and E8s LIBERO arm ran with the empty camera attended** — the E8s *paired* comparisons are unaffected
(both arms share it), but absolute success rates are for that input, not the training-time one (question for the
research side in `STATUS.md`); (iii) `libero_eval.py --empty-camera masked` now leaves the key out, and the nopad
LIBERO pair is re-run with it (below).

## 2. GR00T N1.7 `keep1`, `kvonce`, `keep1+kvonce` (`results/20260926_E14_groot_n17_savings_MAXN`)

`nvidia/GR00T-N1.7-3B` @ `2fc962b9`, Isaac-GR00T @ `376ba890`, best committed rung `vlm_compile` (action head
`reduce-overhead`, language model `max-autotune-no-cudagraphs`, TF32 on), N = 4, as `reports/E3_E4_groot.md`.
- `keep1`: the backbone's `lm_head` ran over every prefix token and its logits were discarded; `logits_to_keep=1`
  runs it on the last token only (`bench/groot_savings.py apply_keep1`).
- `kvonce`: the DiT's 16 cross-attention blocks projected the (per-frame constant) VLM features to K / V at every
  denoising step; computed once per frame, packed, with `to_k` / `to_v` replaced by exact slices
  (`apply_kvonce`: `get_action_with_features` and the DiT forward mirrored).

### Time (50 timed after 20 warm-up each, one process per row, run back to back 21:02–21:08 local)

| row | e2e median [p5–p95] | vision | backbone (vision incl.) | vlm prefix excl. vision | action head |
|---|---|---:|---:|---:|---:|
| `groot_n17_vlm_compile` | **237.01** [236.22–238.15] | 70.25 | 154.14 | 83.84 | 80.62 |
| `+keep1` | **228.89** [227.64–238.87] | 70.40 | 146.19 | 75.75 | 80.39 |
| `+kvonce` | **228.18** [227.70–228.87] | 70.32 | 153.77 | 83.45 | 72.21 |
| `+keep1+kvonce` | **219.98** [219.71–220.34] | 70.27 | 145.49 | 75.22 | 72.28 |
| `groot_n17_vlm_compile` (repeat) | **236.55** [236.22–236.86] | 70.41 | 154.09 | 83.66 | 80.21 |

`keep1` saves **8.1 ms** e2e (backbone −7.95 ms, −5.2 %), `kvonce` **8.8 ms** e2e (action head −8.41 ms,
−10.4 %), together **17.0 ms** (−7.2 % e2e): the savings add. The two references bracket the variants within 0.5 ms.

### DRAM bytes per frame (`…/bytes`; E12b metric set, `--graph-profiling graph`, one inference per variant)

| part (NVTX range) | ref reads / writes ≤ (GB) | `+keep1` | `+kvonce` | `+keep1+kvonce` |
|---|---|---|---|---|
| vision (Qwen3-VL image encoder) | 4.31 / 2.31 | 4.31 / 2.31 | 4.30 / 2.31 | 4.30 / 2.31 |
| vlm prefix (language model) | 3.52 / 0.71 | 3.53 / 0.71 | 3.53 / 0.71 | 3.53 / 0.71 |
| other in the frame (`lm_head`, projector, glue) | **2.00** / 0.11 | **0.65** / 0.03 | 1.98 / 0.11 | **0.65** / 0.03 |
| action head (4 steps; 3 graph workloads) | **11.14** / 0.63 | 11.13 / 0.63 | **9.84** / 0.66 | **9.84** / 0.66 |
| **frame total** | **20.96 / 3.77** | **19.62 / 3.69** | **19.66 / 3.79** | **18.32 / 3.71** |
| frame reads vs ref | 1 | **0.936** | **0.938** | **0.874** |

Each change removes bytes only where it acts: `keep1` 1.35 GB in the rest-of-frame range holding `lm_head`, `kvonce`
1.30 GB in the action head; together 2.64 GB of reads (−12.6 %), additive within 0.01 GB. Writes barely move. A GR00T
frame is 1 643 profiled workloads (vision 796, language model 787 non-graph kernels), so it was captured in chunks of
≤ 450 per process (ncu dies after 480 on this device); the observation values are random (bytes do not depend on
them). Profiled kernel time under ncu (~700 ms per frame) is not latency — use the timing table above.

### Equivalence (`actions_*`, `equiv.csv`, `equiv_summary.json`; 50 inferences, `torch.manual_seed(5000 + i)`, one seeded observation)

| pair | bitwise | MAXABS | REL |
|---|---:|---:|---:|
| control: compiled ref vs compiled ref, new process | **50 / 50** | 0 | 0 |
| control: eager ref vs eager ref, new process | **50 / 50** | 0 | 0 |
| compiled ref vs `+keep1` / `+kvonce` / `+keep1+kvonce` | **50 / 50** each | 0 | 0 |
| eager ref vs `+keep1` / `+kvonce` / `+keep1+kvonce` | **50 / 50** each | 0 | 0 |
| eager ref vs compiled ref | 0 / 50 | 6.75 | 1.27 |

**Row 7: bitwise.** All three variants reproduce the reference's actions exactly, compiled and eager, and the
harness is deterministic run to run. Eager vs compiled differs by REL 1.27 — the size of the difference between two
noise seeds (1.23): the two paths draw different noise from the same seed, so no eager-vs-compiled floor exists for
GR00T here. Every compiled GR00T run (these, the timings, E3/E4/E7) logs a `torch._dynamo` recompile-limit warning on
transformers' `wrapped_forward`; it is the same configuration throughout.

The first ten action sets were invalid and are kept as `invalid_unseeded_inputs/` (see its `NOTE.md`): the
observation's pixels and state come from `np.random`, unseeded, so each process ran a different observation and
every cross-process comparison compared different inputs. Fixed (`--save-actions` seeds `np.random` and hashes the
observation; the comparison refuses unequal hashes) and re-run.

## 3. Not measured

- A GR00T closed-loop (LIBERO-style) comparison: not needed (row 7 bitwise) and no GR00T closed-loop harness exists here.
- Per-kernel DRAM bytes for the variants (T016 1d tooling): graph-level totals per part were taken instead.
- π0 `nopad` in the E12b `graphs_both` harness *time* (the bytes used it; the time used the rung path): T017 measures
  all four padding variants in that harness.

## Reproduce

```
~/venvs/pi0/bin/python bench/e14_pi0_nopad.py --mode time --frames 30 --outdir results/<date>_E14_pi0_nopad_MAXN/time
~/venvs/pi0/bin/python bench/e14_pi0_nopad.py --mode equiv --inputs 50 --outdir results/<date>_E14_pi0_nopad_MAXN/equiv
bash bench/run_ncu_e14.sh results/<date>_E14_pi0_nopad_MAXN/bytes pad,nopad 5      # after one plain warm-up run
bench/groot_phases.py --checkpoint nvidia/GR00T-N1.7-3B --version n17 --steps 4 --warmup 20 --timed 50 --tf32 on \
    --rung vlm_compile --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs [--keep1] [--kvonce]
bash bench/run_ncu_e14_groot.sh results/<date>_E14_groot_n17_savings_MAXN/bytes
python bench/e14_groot_equiv.py results/<date>_E14_groot_n17_savings_MAXN
```
