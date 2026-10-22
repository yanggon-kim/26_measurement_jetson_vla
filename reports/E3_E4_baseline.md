# E3 / E4: eager baseline latency of all six models on Thor, and π0's stock compile modes (T004)

Jetson AGX Thor, MAXN (id 0), GPU 1575 MHz locked, EMC 4266 MHz, `jetson_clocks` + fan, 2026-10-22. Conditions:
- The protocol §1 gate passed before every N block, including GPU idle from the system-wide `nvidia-smi` utilisation.
- Every block started and ended in the **normal** memory-path state (`bench/gate.memory_state_probe()`, T003).
- No soctherm over-current event occurred in any timed block.
- Batch 1 on the Orin's fixtures:
  - π0 and π0.5: one LIBERO frame, 2 cameras plus the empty third camera slot the policy pads with a blank image
    (3 images, P = 816 / 968 tokens);
  - SmolVLA: the same frame, 2 images (P = 177);
  - GR00T: the checkpoint's own keys and shapes, with seeded random pixels; N1.7 uses DROID with 2 cameras × 2 frames
    (P = 273).
- 20 warm-up / 100 timed inferences per N. Medians, with p5–p95 in each directory's `latency_summary.csv`.

**This report was revised after the T004 independent review.** GR00T's regions were redefined so that all three
versions mean the same thing (`vlm_prefix` = the LLM decoder stack; the unused vocabulary projection is its own
`backbone_lm_head` region), and all GR00T runs were repeated (`_v2` directories). The first GR00T directories carry
`SUPERSEDED.md` and are not cited for numbers.

## The answer first: the `eager_fp32` breakdown at each model's default N

| model | N | e2e ms | vision ms (share) | VLM prefix ms (share) | **action expert** ms (share) | other ms | ms / step (instrumented / slope) | gap ÷ floor: vision / prefix / step | Orin eager e2e (vision / prefix / expert, AE %) | Orin ÷ Thor |
|---|---|---|---|---|---|---|---|---|---|---|
| π0 | 10 | 425.9 | 175.9 (41.3 %) | 120.4 (28.3 %) | **126.6 (29.7 %)** | 3.0 | 12.61 / 12.53 | 1.71 / 4.69 / 4.97 | 1069.0 (241.0 / 234.6 / 586.9, 54.9 %) | 2.51 |
| π0.5 | 10 | 1433.1 | 171.3 (12.0 %) | 854.5 (59.6 %) | **404.5 (28.2 %)** | 2.9 | 40.37 / 40.32 | 1.67 / 1.38 / 5.93 | 3028.6 (239.4 / 2162.5 / 617.7, 20.4 %) | 2.11 |
| SmolVLA | 10 | 167.8 | 15.0 (8.9 %) | 14.1 (8.4 %) | **135.3 (80.6 %)** | 3.4 | 13.44 / 13.39 | 4.53 / 11.57 / 16.48 | 727.5 (33.3 / 65.8 / 620.6, 85.3 %) | 4.33 |
| GR00T N1.7 | 4 | 155.3 | 28.9 (18.6 %) | 25.2 (16.2 %) | **74.2 (47.8 %)** † | 27.0 | 17.71 / 17.70 | 5.37 / 4.03 / 2.08 | 372.5 (70.6 / 88.6 / 211.1, 56.7 %) ‡ | 2.74 ‡ |
| GR00T N1.6 | 4 | 135.0 | 21.2 (15.7 %) | 20.5 (15.2 %) | **73.2 (54.2 %)** | 20.2 | 18.26 / 18.20 | 3.25 / 3.28 / 2.13 | 327.2 (61.9 / 61.7 / 201.3, 61.5 %) ‡ | 2.72 ‡ |
| GR00T N1.5 | 4 | 141.8 | 19.0 (13.4 %) | 31.1 (21.9 %) | **52.6 (37.1 %)** | 39.1 | 11.18 / 11.20 | 3.71 / 3.97 / 2.57 | 255.2 (50.2 / 100.6 / 98.5, 38.6 %) ‡ | 2.05 ‡ |

† GR00T N1.7's expert time **varies from process to process**: the medians of nine runs at N = 4 span 72.2–85.7 ms
(Anomalies 1). 74.2 ms is the v2 sweep's median; the superseded first sweep measured 83.9 ms.

Spread and conditions of the same runs (p5–p95 over n = 100 timed inferences each; tokens from `tokens.json`; peak
memory and control frequency from `raw/per_N_<N>.json`):

| model | e2e p5–p95 | vision p5–p95 | prefix p5–p95 | expert p5–p95 | P processed / valid | S | images | peak GB allocated | Hz (timed loop) |
|---|---|---|---|---|---|---|---|---|---|
| π0 | 423.6–442.8 | 175.8–176.0 | 120.0–120.9 | 124.4–143.2 | 816 / 533 | 51 | 3 | 9.19 | 2.33 |
| π0.5 | 1431.8–1436.9 | 171.2–171.4 | 854.1–855.0 | 403.3–408.1 | 968 / 574 | 50 | 3 | 17.05 | 0.70 |
| SmolVLA | 167.0–169.1 | 14.9–15.2 | 13.9–14.4 | 134.6–136.6 | 177 / 151 | 50 | 2 | 1.12 | 5.93 |
| GR00T N1.7 | 154.3–161.4 | 28.8–29.9 | 25.0–26.6 | 74.0–77.8 | 273 / 273 | 41 | 4 (2 cameras × 2 frames) | 6.62 | 6.39 |
| GR00T N1.6 | 134.5–136.9 | 21.1–21.4 | 20.3–20.9 | 73.1–73.5 | 284 / 284 | 51 | per checkpoint | 6.96 | 7.38 |
| GR00T N1.5 | 133.6–143.3 | 18.9–19.1 | 30.9–31.2 | 52.4–52.8 | 814 / 814 | 49 | per checkpoint | 5.93 | 7.07 |

- P: the prefix tokens the LLM processes, and how many are real. LeRobot pads the language and empty-camera tokens, so
  its valid counts are lower; GR00T processes no padding.
- S: the action tokens the expert processes per step. π0 has 51 = 50 actions + 1 state token. GR00T's
  `action_horizon` is 40 / 16 / 16; S is measured on the DiT input.

‡ **The Orin's GR00T e2e is a different quantity from Thor's e2e column.**
- On the Orin it is the model call `model.get_action` on inputs collated beforehand: the GPU path, without the
  policy's CPU processing (`reference/orin/bench/groot_phases.py:631-635`).
- Its AE share and the Orin ÷ Thor ratio are therefore compared with Thor's `model_get_action`, in the table below.
- Thor's e2e column is `Gr00tPolicy.get_action`, which adds the CPU processing.

Columns:
- "ms / step, instrumented" = mean of the per-step medians at the default N; "slope" = b of `ae_total = a + b·N` over
  N ∈ {1, 2, 4, 6, 8, 10}.
- "other" = e2e − vision − prefix − expert. For GR00T it holds:
  - the unused `lm_head`: 3.5 / 3.9 / 19.9 ms for N1.7 / N1.6 / N1.5;
  - the CPU processing of `Gr00tPolicy.get_action` outside the model: 19.3 / 14.7 / 17.4 ms;
  - the rest: 4.2 / 1.6 / 1.9 ms. For N1.7, 2.9 ms of it is inside the backbone, mostly CPU-side `get_rope_index`
    (Region split, below).

Sources:
- Thor: `results/20261022_E3_{pi0,pi05,smolvla}_eager_Nsweep/` and `results/20261022_E3_groot_n1{7,6,5}_eager_Nsweep_v2/`
  (`latency_summary.csv`, `step_fit.csv`, `floor.csv`).
- Orin: `reference/orin/reports/E4d_phase_balance.md`, the `eager_fp32` rows (lines 41, 62, 84, 106, 120, 136).
  - The same values are in `reference/orin/reports/E4i_best_combinations.md` lines 25–30 and `docs/00_context.md`.
  - `reference/orin/reports/E3_E4_models.md:55` labels π0's 1069.04 ms differently; E4d is unambiguous.

**GR00T against the Orin, like for like.** Two definitions differ, so both are aligned here:
- e2e: the model call on both devices (‡ above).
- Prefix: the Orin derived GR00T's prefix as backbone − vision, which includes the unused `lm_head` and the glue, so
  the matching Thor number is `vlm_backbone − vision`.

| model | Thor model call ms | vision (share) | Thor decoder-stack prefix | Thor backbone − vision (Orin definition) | expert (share of the model call) | Orin: e2e / prefix / AE share | Orin ÷ Thor: e2e / vision / prefix (Orin def.) / expert |
|---|---|---|---|---|---|---|---|
| GR00T N1.7 | 135.8 | 28.9 (21.3 %) | 25.2 | 31.5 | 74.2 (54.7 %) | 372.5 / 88.6 / 56.7 % | 2.74 / 2.44 / 2.81 / 2.84 |
| GR00T N1.6 | 120.3 | 21.2 (17.6 %) | 20.5 | 24.8 | 73.2 (60.9 %) | 327.2 / 61.7 / 61.5 % | 2.72 / 2.93 / 2.49 / 2.75 |
| GR00T N1.5 | 124.4 | 19.0 (15.3 %) | 31.1 | 51.3 | 52.6 (42.3 %) | 255.2 / 100.6 / 38.6 % | 2.05 / 2.64 / 1.96 / 1.87 |

Thor's model call also holds 7.4 / 5.4 / 21.7 ms outside the three parts. Most of it is the unused `lm_head`
(3.5 / 3.9 / 19.9 ms); the rest is glue and position-id work.

**What it says:**
- Thor runs every model 2.1–4.3× faster than the Orin in eager mode (GR00T: model call against model call).
- The action expert's share relative to the Orin, at each model's default N:
  - **falls** for π0 (54.9 → 29.7 %) and SmolVLA (85.3 → 80.6 %);
  - **stays about the same** for GR00T N1.7 (56.7 → 54.7 % of the model call; 53.9–57.8 % across the seven N1.7
    runs that record the model call, Anomalies 1) and N1.6 (61.5 → 60.9 %);
  - **rises** for π0.5 (20.4 → 28.2 %) and GR00T N1.5 (38.6 → 42.3 %).
- Per part, Orin ÷ Thor (eager):

  | model | vision | prefix | expert |
  |---|---|---|---|
  | π0 | 1.37 | 1.95 | **4.64** |
  | π0.5 | 1.40 | **2.53** | 1.53 |
  | SmolVLA | 2.22 | **4.67** | 4.59 |

  (GR00T: in the table above.) Only π0's expert speeds up clearly more than its prefix. π0.5's expert speeds up least
  of its three parts. SmolVLA's prefix and expert speed up alike.
- Whether the expert dominates after optimisation is T005's question. These are the baselines it starts from.

**Second row, the shipped eager path.** It uses torch default flags, so it differs from `eager_fp32` only in cuDNN
TF32 being on. Sources: `results/20261022_E3_{pi0,pi05,smolvla}_shipped_N10/` and
`results/20261022_E3_groot_n1{7,6,5}_shipped_N4_v2/`.

| model | e2e | vision | prefix | expert (share) | vs `eager_fp32` |
|---|---|---|---|---|---|
| π0 | 429.8 | 175.1 | 126.0 | 125.9 (29.3 %) | e2e +0.9 %; prefix **+4.6 %** |
| π0.5 | 1438.4 | 174.7 | 854.1 | 406.6 (28.3 %) | e2e +0.4 % |
| SmolVLA | 171.8 | 15.1 | 14.3 | 138.8 (80.8 %) | e2e +2.4 %; expert +2.6 % |
| GR00T N1.7 | 151.6 | 28.9 | 25.1 | 72.8 (48.0 %) | e2e −2.4 %; expert −1.9 % (within the run-to-run spread) |
| GR00T N1.6 | 135.3 | 21.2 | 20.6 | 73.4 (54.3 %) | e2e +0.2 % |
| GR00T N1.5 | 140.6 | 18.9 | 31.2 | 52.6 (37.4 %) | e2e −0.8 % |

- Every part matches `eager_fp32` within 3 % except π0's prefix, which is 4.6 % slower on the shipped path.
- N1.5's `Gr00tPolicy.get_action` wraps inference in `torch.autocast(bf16)` in both rows (`gr00t/model/policy.py:190`).

## Per-step time, both definitions (protocol §2)
- The mean of the per-step medians and the slope agree within 0.7 % for every model.
- r² of the linear fit: π0 0.99796, SmolVLA 0.99838, all others ≥ 0.99998.
- Steps are equal except step 0 in the LeRobot models, which is slower than the later steps (first-call effects):
  π0 +0.8–1.0 ms, π0.5 +0.7–0.9 ms, SmolVLA +0.3–0.4 ms. The GR00T steps agree within 0.1 ms.
- Intercepts `a`: 0.2–7.8 ms. N1.5's 7.8 ms and N1.7's 3.5 ms hold the per-call prologue inside `ae_total` (state
  encoder, layer norm over the VLM features) before the first step.

## Floors (protocol §9; `floor.csv`, measured ceilings of `results/20261022_E1_peak/hw_constants.json`)
How the floors are built:
- Bytes and FLOPs come from one accounting pass of the real model on the real input (`accounting.json`).
- FLOPs come from torch `FlopCounterMode` per region, with formulas registered for `flash_attn`'s custom ops (GR00T).
  - The counted ops are listed per region in `flop_ops`.
  - For the LeRobot models that list is in `results/20261022_E3_lerobot_accounting_flop_ops/`; its totals equal the
    E3 runs' exactly.
- Bytes are the parameters each region touches, plus the hand-over it reads:
  - embeddings: only the rows looked up;
  - GR00T's `CategorySpecificLinear`: only the slice of the embodiment actually used;
  - hand-over: LeRobot reads the KV-cache tensors the prefill returns; GR00T reads the DiT's `encoder_hidden_states`.
- The ceiling is the measured one for the part's dominant parameter dtype. π0's and π0.5's vision towers, and all of
  π0.5, are fp32: at `highest` they run on SIMT FP32 (6.43 TFLOP/s), so their compute floors are large.
- `ae_total` counts the expert's weights once. `ae_total_restream` counts them once per step, because they exceed the
  32 MiB L2; it is compared with the sum of the per-step medians.

| model | vision floor ms (bound) | prefix floor ms (bound) | one-step floor ms (bound; bytes) | gap: vision / prefix / step / restream |
|---|---|---|---|---|
| π0 | 102.8 (compute, fp32) | 25.7 (compute, bf16) | 2.52 (memory; 0.65 GB) | 1.71 / 4.69 / 4.97 / 5.00 |
| π0.5 | 102.8 (compute, fp32) | 618.5 (compute, fp32) | 6.80 (memory; 1.76 GB) | 1.67 / 1.38 / 5.93 / 5.94 |
| SmolVLA | 3.32 (compute) | 1.22 (memory) | 0.81 (memory; 0.21 GB) | 4.53 / 11.57 / 16.48 / 16.55 |
| GR00T N1.7 | 5.39 (compute) | 6.24 (memory) | 8.53 (memory; 2.20 GB) | 5.37 / 4.03 / 2.08 / 2.08 |
| GR00T N1.6 | 6.52 (compute) | 6.24 (memory) | 8.53 (memory; 2.20 GB) | 3.25 / 3.28 / 2.13 / 2.14 |
| GR00T N1.5 | 5.12 (compute) | 7.83 (compute) | 4.35 (memory; 1.12 GB) | 3.71 / 3.97 / 2.57 / 2.57 |

Reading:
- The **GR00T DiT steps run 2.1–2.6× above their memory floor.** They stream 1.1–2.2 GB of weights per step. Of all
  the parts, they come closest to a pure weight-streaming workload, the kind PIM would target.
- The **LeRobot experts are 5–17× above their floors**.
  - π0's 126 projection GEMMs per step (7 per layer × 18 layers) took 3.29 ms in E2, at 1.37× their floor.
  - The other ≈ 9 ms of π0's 12.5 ms step is outside them: attention, norms, RoPE, KV-cache copies and kernel
    launches. T006 / T008 decompose it.
- SmolVLA's step (13.4 ms against a 0.81 ms floor) is dominated by its many small kernels.

## Region split (T004 step 2)
- Vision and prefix kernel sets are **disjoint for all six models**, and their intervals never overlap (nsys,
  2 inferences each).
  - LeRobot: `results/20261022_E3_region_split_check/`.
  - GR00T: `results/20261022_E3_groot_region_split_check/`.
- GR00T, timing check (E3 v2, N = 4, per-iteration ratios, median over 100 iterations):

| model | (vision + prefix) ÷ backbone | (vision + prefix + `lm_head`) ÷ backbone | same, shipped path | unused `lm_head` ms | rest of the backbone: CUDA events (per-iteration median) | of which GPU kernels (nsys, per inference) |
|---|---|---|---|---|---|---|
| N1.7 | 0.895 | **0.952** | 0.952 | 3.46 | 2.88 ms | 0.25 ms of index / scatter kernels; the rest is CPU-side `get_rope_index` |
| N1.6 | 0.907 | **0.992** | 0.992 | 3.92 | 0.37 ms | 0.06 ms |
| N1.5 | 0.712 | **0.995** | 0.994 | 19.88 | 0.38 ms | 0.11 ms |

- **The shipped GR00T code computes logits it throws away.**
  - Inside `backbone.forward`, the language model's `lm_head` projects every prefix token onto the vocabulary
    (N1.7: 273 tokens × 151,936 entries, 169.9 GFLOP).
  - GR00T keeps only `hidden_states[-1]`.
  - Cost: 3.5 / 3.9 / 19.9 ms per inference (N1.7 / N1.6 / N1.5), which is 2.2 / 2.9 / 14.0 % of e2e.
  - It belongs to no part of the action pipeline and is reported in "other".
- With `lm_head` named, N1.6 and N1.5 sum to the backbone within 2 %.
- **N1.7 still leaves 4.8 % (≈ 2.9 ms) unaccounted**, which fails the 2 % rule.
  - Its GPU part is the 0.25 ms of glue kernels.
  - The rest is CPU-side multimodal position-id work, with the GPU idle.
  - Named, not fixed (protocol §2).

## E4: π0 under PyTorch's stock compile modes (`results/20261022_E4_pi0_compile_modes/`, matmul `highest`, TF32 off)

| target | mode | compile s | graph breaks (unique / log lines) | e2e N = 10 ms | ms / step (slope) | intercept ms |
|---|---|---|---|---|---|---|
| expert (`denoise_step`) | `default` | 22.3 | 1 / 33 | 391.9 | 8.67 | 305.2 |
| | `reduce-overhead` | 21.0 | 1 / 33 | 380.1 | 8.29 | 297.3 |
| | `max-autotune` | 44.5 | 1 / 33 | 383.5 | 8.04 | 303.2 |
| | `max-autotune-no-cudagraphs` | 38.8 | 1 / 33 | 392.6 | 9.20 | 300.6 |
| whole (`sample_actions`) | `default` | 82.8 | 3 / 68 | 368.2 | 8.78 | 280.7 |
| | `reduce-overhead` | 85.1 | 3 / 68 | 362.2 | 8.40 | 278.2 |
| | `max-autotune` | 164.4 | 3 / 68 | **360.7** | 8.44 | 276.5 |
| | `max-autotune-no-cudagraphs` | 152.5 | 3 / 68 | 376.6 | 8.83 | 288.3 |

Eager: 425.9 ms, 12.53 ms per step.
- **No recompilation** at any N.
  - The evidence is 0 `[__recompiles]` lines in the `TORCH_LOGS=recompiles` stream of every configuration.
  - torch 2.10 does not populate `counters["recompiles"]`, so that counter is not used.
- Every configuration graph-breaks at `copy.deepcopy(past_key_values)` inside `denoise_step` (`modeling_pi0.py:919`).
  - So the loop stays in Python, and each step's graph is reused.
  - "Log lines" counts every `[__graph_breaks]` record, including repeats of the same break on each call during
    warm-up.
- Inductor logs **"Not enough SMs to use max_autotune_gemm mode"** in every mode (threshold 68 SMs vs this GPU's 20).
  So `max-autotune` does not autotune GEMMs.
- **CUDA graphs were used** in the two graph modes (`reduce-overhead`, `max-autotune`): Inductor logged 0
  "skipping cudagraphs" messages in every configuration (`cudagraph_skips` in `compile_modes.csv`).
- Peak memory: 9.07–9.19 GB. Actions differ from eager by 3.2e-3 to 4.9e-3 (relative L2, same seed).
- Stock compilation buys at most 15 % end to end.
  - The intercepts (276.5–305.2 ms, mostly vision and prefix) barely move.
  - T005's ladder separates the parts.

## Anomalies and confidence
1. **GR00T N1.7's expert time varies between processes** (`results/20261022_E3_groot_n17_state_vs_latency/`).
   - Per-call `ae_total` at N = 4 clusters near ≈ 72, ≈ 77 and ≈ 81–88 ms.
   - Within one process the samples sit mostly in one or two clusters; the dominant cluster changes from process to
     process. Medians of nine runs: 72.2–85.7 ms.
   - The superseded first sweep sat in the slow cluster at **every** N: step 0 took 18.4–18.7 ms and steps 1 onwards
     20.1–21.1 ms (slope 20.35 ms per step).
   - The v2 sweep sat in the fast cluster at every N (≈ 18 ms per step for every step; at N = 10, 100 of 100
     iterations).
   - **Precision setting correlates, but does not decide it.**
     - All four shipped-path runs (cuDNN TF32 on) sat in the low clusters (medians 72.2–74.0 ms).
     - Of the five `eager_fp32` runs, three sat in the high cluster (83.9–85.7 ms), the per-iteration-probe run at
       ≈ 77 ms, and the v2 sweep in the low cluster (74.2 ms).
   - It is **not** the T003 memory-path state: the per-iteration probe read "normal" in every cluster.
   - The cause is not established. Because the state persists for a whole process, per-process choices are
     candidates: kernel selection (the cuDNN / cuBLAS heuristics differ between the two precision settings) and memory
     placement. T006's nsys traces compare a fast and a slow process.
   - Consequence: N1.7's expert time carries a run-to-run uncertainty of the size of that range. Its measured AE share
     is 47.4–51.0 % of e2e over the nine runs, and 53.9–57.8 % of the model call over the seven runs that record it.
2. **π0's eager expert has a slow tail.**
   - Its p95 is 143.2 ms against a median of 126.6 ms (p95/p5 1.15).
   - At the default N, every other vision, prefix and expert region has p95/p5 ≤ 1.07.
   - At small N the GR00T N1.7 regions spread more (N = 1: prefix 1.10, `ae_total` 1.10).
   - GR00T's CPU-side `outside_model` time varies more (N1.5 p5–p95 9.0–18.7 ms), which widens N1.5's e2e p5–p95 to
     133.6–143.3 ms.
3. **The floors assume every parameter touched is read once per call (once per step for `restream`)**, and they ignore
   activations.
   - The vision floors are compute floors at the measured burst ceilings.
   - Under back-to-back load the sustained bf16 rate is 104 TFLOP/s (T003).
4. **The `ae_total` floor** (weights read once) is shown for the protocol; the `ae_total_restream` row (weights once
   per step) is the physical one.

Confidence:
- High: every e2e and region median except N1.7's expert (see 1), the per-step agreement, and the region disjointness.
- Medium: the floors. Their FLOP and byte accounting is from executed ops; ops without a FLOP formula would be
  missing, and `flop_ops` lists what was counted.

## Reproduce
```bash
source env/thor_env.sh
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/run_e3.py --model pi0 \
  --out results/<date>_E3_pi0_eager_Nsweep --hw results/20261022_E1_peak/hw_constants.json        # pi05, smolvla alike
(cd /tmp && env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=1 ~/venvs/groot17/bin/python <repo>/bench/run_e3.py \
  --model groot_n17 --out <repo>/results/<dir> --hw <repo>/results/20261022_E1_peak/hw_constants.json)  # n16 / n15: their venvs
... run_e3.py --model <m> --precision shipped --steps <N> --out results/<date>_E3_<m>_shipped_N<N>
... run_e3.py --model <m> --accounting-only --out <dir>                                       # accounting pass alone
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o scratch/<run>/<m> \
  <venv>/bin/python bench/nsys_regions_target.py --model <m> --inferences 2
~/venvs/base/bin/python bench/region_split_check.py --nsys-prefix results/<split>/nsys/<m> --e3 results/<E3 dir> --model <m> --n <N> --out results/<split>
E4_CACHE_ROOT=scratch/t004/e4cache HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/run_e4.py --all --out results/<date>_E4_pi0_compile_modes
```
Each directory's `meta.json` holds its exact command and bench commit.

## Deviations
- **LeRobot e2e.** It is the model call (`predict_action_chunk`, as on the Orin). Preprocess (0.5–0.9 ms) and
  postprocess (0.1 ms) are timed separately in the same iteration (`e2e_full_wall`).
- **GR00T e2e.** It is `Gr00tPolicy.get_action`, which includes its CPU processing (`outside_model`, 14.7–19.3 ms).
- **GR00T inputs.** Seeded random pixels at the checkpoint's shapes (as on the Orin), not a real frame.
- **GR00T regions.** `vlm_prefix` is the decoder stack and `backbone_lm_head` is separate, so Thor's GR00T prefix is
  not the Orin's (see the comparison table above).
- **Floors.** From executed-op accounting rather than per-architecture formulas.
- **E4.** At N ∈ {2, 6, 10} (slope from 3 points) and at matmul `highest`.
- **Re-runs after the review.**
  - All GR00T E3 runs: `_v2` directories.
  - The GR00T split check: `results/20261022_E3_groot_region_split_check/`.
  - The LeRobot accounting, to add `flop_ops`.
- **Split validation per mode (protocol §2).** The nsys disjointness check ran on `eager_fp32` only. The shipped path
  uses the same region code, and its GR00T timing sums were checked from the shipped runs' samples (table in Region
  split). The LeRobot shipped rows have no separate nsys check.
- **Added beyond the task.** The shipped-path rows (at the default N only) and the N1.7 variation check.

## Self-review (protocol §11)
| check | result |
|---|---|
| six models, E3 at all six N, separate vision and prefix | yes |
| GR00T split validated | yes: disjoint for all; with the unused `lm_head` named, N1.6 / N1.5 sum within 2 %; N1.7 at 0.952, the remaining ≈ 2.9 ms identified (CPU-side position ids, 0.25 ms glue kernels) |
| P processed and valid, S, peak memory, control frequency | yes (table under the lead table) |
| `floor.csv` per model against T003's ceilings | yes |
| π0's four stock modes: compile time, graph breaks, ms per step | yes (+ recompiles from `TORCH_LOGS`, CUDA graphs, SM message, peak memory) |
| `meta.json` complete, raw files committed, secrets scan clean, no file > 50 MB | yes (23 directories: 22 `E3_*` + 1 `E4_*`; 7 superseded) |
| sizes in `STATUS.md`, `PROGRESS.md` updated | at the close |
| independent review (P0) | done; findings and fixes in the T004 status block |
