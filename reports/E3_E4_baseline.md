# E3 / E4: eager baseline latency of all six models on Thor, and π0's stock compile modes (T004)

Jetson AGX Thor, MAXN (id 0), GPU 1575 MHz locked, EMC 4266 MHz, `jetson_clocks` + fan, 2026-10-22. Conditions:
- The protocol §1 gate passed before every N block, including GPU idle from the system-wide `nvidia-smi` utilisation.
- Every block started and ended in the **normal** memory-path state (`bench/gate.memory_state_probe()`, T003).
- No soctherm over-current event occurred in any timed block.
- Batch 1 on the Orin's fixtures:
  - LeRobot models: one LIBERO frame, 2 cameras.
  - GR00T: the checkpoint's own keys and shapes, with seeded random pixels; N1.7 uses DROID with 2 cameras × 2 frames.
- 20 warm-up / 100 timed inferences per N. Medians, with p5–p95 in each directory's `latency_summary.csv`.

## The answer first: the `eager_fp32` breakdown at each model's default N

| model | N | e2e ms | vision ms (share) | VLM prefix ms (share) | **action expert** ms (share) | other ms | ms / step (instrumented / slope) | gap ÷ floor: vision / prefix / step | Orin eager e2e (vision / prefix / expert, AE %) | Orin ÷ Thor |
|---|---|---|---|---|---|---|---|---|---|---|
| π0 | 10 | 425.9 | 175.9 (41.3 %) | 120.4 (28.3 %) | **126.6 (29.7 %)** | 3.0 | 12.61 / 12.53 | 1.71 / 4.69 / 4.97 | 1069.0 (241.0 / 234.6 / 586.9, 54.9 %) | 2.51 |
| π0.5 | 10 | 1433.1 | 171.3 (12.0 %) | 854.5 (59.6 %) | **404.5 (28.2 %)** | 2.9 | 40.37 / 40.32 | 1.67 / 1.38 / 5.93 | 3028.6 (239.4 / 2162.5 / 617.7, 20.4 %) | 2.11 |
| SmolVLA | 10 | 167.8 | 15.0 (8.9 %) | 14.1 (8.4 %) | **135.3 (80.6 %)** | 3.4 | 13.44 / 13.39 | 4.53 / 11.57 / 16.48 | 727.5 (33.3 / 65.8 / 620.6, 85.3 %) | 4.33 |
| GR00T N1.7 | 4 | 166.2 | 29.3 (17.6 %) | 26.0 (15.6 %) | **83.9 (50.5 %)** † | 27.0 | 20.09 / 20.35 | 5.43 / 4.17 / 1.93 | 372.5 (70.6 / 88.6 / 211.1, 56.7 %) | 2.24 |
| GR00T N1.6 | 4 | 136.2 | 21.0 (15.4 %) | 24.9 (18.3 %) | **74.6 (54.8 %)** | 15.7 | 18.61 / 18.60 | 3.22 / 2.88 / 1.76 | 327.2 (61.9 / 61.7 / 201.3, 61.5 %) | 2.40 |
| GR00T N1.5 | 4 | 143.2 | 19.1 (13.3 %) | 51.6 (36.0 %) | **53.3 (37.2 %)** | 19.1 | 11.37 / 11.41 | 3.73 / 4.40 / 1.80 | 255.2 (50.2 / 100.6 / 98.5, 38.6 %) | 1.78 |

† GR00T N1.7's expert time is **bimodal**: ≈ 72 or ≈ 85 ms per call, with run-dependent shares (see Anomalies). 83.9 ms
is the median of a run with 0 of 100 iterations in the fast mode; the shipped-path run has 96 of 100 there (72.2 ms,
47.7 %).

Sources:
- Thor: `results/20261022_E3_<model>_eager_Nsweep/` (`latency_summary.csv`, `step_fit.csv`, `floor.csv`).
- Orin: `docs/00_context.md` ("Orin numbers"), from `reference/orin/reports/E4i_best_combinations.md`.

**What it says:**
- Thor runs every model 1.8–4.3× faster than the Orin in eager mode.
- The action expert's share **falls** relative to the Orin for five of six models:
  - π0 54.9 → 29.7 %;
  - SmolVLA 85.3 → 80.6 %;
  - N1.7 56.7 → 50.5 % (47.7 % in the fast mode);
  - N1.6 61.5 → 54.8 %;
  - N1.5 38.6 → 37.2 %;
  - π0.5 is the exception: 20.4 → 28.2 %.
- In eager mode the expert speeds up *more* than the prefix for the LeRobot models: π0's expert 586.9 → 126.6 ms
  (4.6×), its prefix 234.6 → 120.4 (1.9×).
- Whether the expert dominates after optimisation is T005's question. These are the baselines it starts from.

**Second row, the shipped eager path** (torch default flags; differs from `eager_fp32` only in cuDNN TF32 on;
`results/20261022_E3_<model>_shipped_N<N>/`):

| model | e2e | vision | prefix | expert (share) |
|---|---|---|---|---|
| π0 | 429.8 | 175.1 | 126.0 | 125.9 (29.3 %) |
| π0.5 | 1438.4 | 174.7 | 854.1 | 406.6 (28.3 %) |
| SmolVLA | 171.8 | 15.1 | 14.3 | 138.8 (80.8 %) |
| GR00T N1.7 | 151.4 | 28.8 | 25.3 | 72.2 (47.7 %) |
| GR00T N1.6 | 136.1 | 21.1 | 24.4 | 74.7 (54.9 %) |
| GR00T N1.5 | 139.6 | 19.0 | 52.0 | 51.1 (36.6 %) |

These match `eager_fp32` within 1–3 % except N1.7 (bimodality, not the flag) and N1.5's expert (−4 %). N1.5's
`Gr00tPolicy.get_action` wraps inference in `torch.autocast(bf16)` in both rows (`gr00t/model/policy.py:190`).

## Per-step time, both definitions (protocol §2)
- The instrumented per-step medians and the slopes of `ae_total = a + b·N` over N ∈ {1, 2, 4, 6, 8, 10} agree within
  1.3 % for every model, with r² ≥ 0.998.
- Steps are equal except step 0, which is 0.4–1.0 ms slower in π0, π0.5 and SmolVLA (first-call effects) and 1.7 ms
  *faster* in N1.7.
- Intercepts are 0.1–7.7 ms (N1.5's includes its per-call prologue).

## Floors (protocol §9; `floor.csv`, measured ceilings of `results/20261022_E1_peak/hw_constants.json`)
How the floors are built:
- Bytes and FLOPs come from one accounting pass of the real model on the real input (`accounting.json`):
  - FLOPs: torch `FlopCounterMode` per region, with formulas registered for `flash_attn`'s custom ops (GR00T);
  - bytes: parameters each region touches (embeddings: rows looked up), plus the hand-over it reads (LeRobot: the
    KV-cache tensors the prefill returns; GR00T: the DiT's `encoder_hidden_states`).
- The ceiling is the measured one for the part's dominant parameter dtype. π0's and π0.5's vision towers, and all of
  π0.5, are fp32: at `highest` they run on SIMT FP32 (6.43 TFLOP/s), so their compute floors are large.
- `ae_total_restream` counts the expert's weights once per step (they exceed the 32 MiB L2); `ae_total` counts them
  once.

| model | vision floor ms (bound) | prefix floor ms (bound) | one-step floor ms (bound) | gap: vision / prefix / step / ae_total_restream |
|---|---|---|---|---|
| π0 | 102.8 (compute, fp32) | 25.7 (compute, bf16) | 2.52 (memory) | 1.71 / 4.69 / 4.97 / 5.02 |
| π0.5 | 102.8 (compute, fp32) | 618.5 (compute, fp32) | 6.80 (memory) | 1.67 / 1.38 / 5.93 / 5.95 |
| SmolVLA | 3.32 (compute) | 1.22 (memory) | 0.81 (memory) | 4.53 / 11.57 / 16.48 / 16.66 |
| GR00T N1.7 | 5.39 (compute) | 6.24 (memory) | 10.56 (memory) | 5.43 / 4.17 / 1.93 / 1.99 |
| GR00T N1.6 | 6.52 (compute) | 8.64 (memory) | 10.56 (memory) | 3.22 / 2.88 / 1.76 / 1.77 |
| GR00T N1.5 | 5.12 (compute) | 11.72 (compute) | 6.33 (memory) | 3.73 / 4.40 / 1.80 / 2.11 |

Reading:
- The **GR00T DiT steps run near their memory floor** (1.76–1.93×). They stream 2.2–2.7 GB of weights per step,
  close to what PIM would target.
- The **LeRobot experts are 5–16× above their floors**. π0's 126 projection GEMMs per step (7 per layer × 18 layers) took 3.29 ms in
  E2, at 1.37× their floor. The other ≈ 9 ms of π0's 12.5 ms step is outside them: attention, norms, RoPE, KV-cache
  copies and kernel launches. T006 / T008 decompose it.
- SmolVLA's step (13.4 ms against a 0.81 ms floor) is dominated by its many small kernels.

## Region split (T004 step 2, `results/20261022_E3_region_split_check/`)
- Vision and prefix kernel sets are **disjoint for all six models** (nsys, 2 inferences each).
- GR00T `vision` + `vlm_prefix` ÷ `vlm_backbone`: N1.6 0.993, N1.5 0.995 (pass the 2 % rule).
- **N1.7: 0.895, which fails the rule, for an identified reason.**
  - Inside `backbone.forward`, Qwen3-VL's `lm_head` computes logits over 273 tokens × 151,936 vocabulary entries:
    169.9 GFLOP, 3.6–3.9 ms.
  - GR00T discards those logits; it keeps only `hidden_states[-1]`.
  - Index and scatter kernels add 0.2 ms, plus CPU-side position-id work.
  - These ≈ 6.6 ms belong to N1.7's "other" (27.0 ms, which also holds 19.4 ms of CPU pre- and post-processing outside
    the model call).
- The Orin derived GR00T's prefix as backbone − vision, so its N1.7 prefix (88.6 ms) **included** the unused logits.
  Thor's 26.0 ms does not. Like for like: 26.0 + 6.6 = 32.6 ms.

## E4: π0 under PyTorch's stock compile modes (`results/20261022_E4_pi0_compile_modes/`, matmul `highest`, TF32 off)

| target | mode | compile s | graph breaks | e2e N = 10 ms | ms / step (slope) |
|---|---|---|---|---|---|
| expert (`denoise_step`) | `default` | 22.3 | 1 | 391.9 | 8.67 |
| | `reduce-overhead` | 21.0 | 1 | 380.1 | 8.29 |
| | `max-autotune` | 44.5 | 1 | 383.5 | 8.04 |
| | `max-autotune-no-cudagraphs` | 38.8 | 1 | 392.6 | 9.20 |
| whole (`sample_actions`) | `default` | 82.8 | 3 | 368.2 | 8.78 |
| | `reduce-overhead` | 85.1 | 3 | 362.2 | 8.40 |
| | `max-autotune` | 164.4 | 3 | **360.7** | 8.44 |
| | `max-autotune-no-cudagraphs` | 152.5 | 3 | 376.6 | 8.83 |

Eager: 425.9 ms, 12.53 ms per step.
- **No recompilation** at any N.
- Every configuration graph-breaks at `copy.deepcopy(past_key_values)` inside `denoise_step` (`modeling_pi0.py:919`),
  so the loop stays in Python and each step's graph is reused.
- Inductor logs **"Not enough SMs to use max_autotune_gemm mode"** in every mode (threshold 68 SMs vs this GPU's 20),
  so `max-autotune` does not autotune GEMMs.
- Peak memory: 9.07–9.19 GB. Actions differ from eager by 3.2e-3 to 4.9e-3 (relative L2, same seed).
- Stock compilation buys at most 15 % end to end: the 280–305 ms of vision and prefix are untouched or barely changed.
  T005's ladder separates the parts.

## Anomalies and confidence
1. **GR00T N1.7's expert time is bimodal**, about 72 vs 85 ms per call
   (`results/20261022_E3_groot_n17_state_vs_latency/`).
   - The share of fast iterations varied between 0 and 96 % across runs, under both precision settings.
   - It is **not** the T003 memory-path state: the per-iteration probe read "normal" in both modes.
   - The cause is not established. The N1.7 AE share is therefore 47.7–50.5 % depending on the mode. T006's nsys
     traces will compare fast and slow iterations.
2. **π0's eager expert has a slow tail** (p95 143.7 ms vs median 126.6 ms); every other region has p95 / p5 ≤ 1.07.
3. **The floors assume all parameters touched are read once**, and they ignore activations. The vision floors are
   compute floors at the measured burst ceilings. Under back-to-back load the sustained bf16 rate is 104 TFLOP/s (T003).
4. **The `ae_total` floor** (weights read once, 42.6× for π0) is shown for the protocol; the `ae_total_restream` row
   (weights once per step) is the physical one.

Confidence:
- High: every e2e and region median (p5–p95 within ±1–3 %), the per-step agreement, and the region disjointness.
- Medium: the floors. Their FLOP and byte accounting is from executed ops, but attention FLOPs for ops without a
  formula would be missing; `flop_ops` in `accounting.json` lists what was counted.

## Reproduce
```bash
source env/thor_env.sh
HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/run_e3.py --model pi0 --out results/<date>_E3_pi0_eager_Nsweep            # pi05, smolvla alike
(cd /tmp && HF_HUB_OFFLINE=1 ~/venvs/groot17/bin/python <repo>/bench/run_e3.py --model groot_n17 --out <repo>/results/<dir>)  # n16 / n15 with their venvs
... run_e3.py --model <m> --precision shipped --steps <N> --out results/<date>_E3_<m>_shipped_N<N>
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o scratch/<run>/<m> \
  <venv>/bin/python bench/nsys_regions_target.py --model <m> --inferences 2
~/venvs/base/bin/python bench/region_split_check.py --nsys-prefix results/<split>/nsys/<m> --e3 results/<E3 dir> --model <m> --n <N> --out results/<split>
E4_CACHE_ROOT=scratch/t004/e4cache HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/run_e4.py --all --out results/<date>_E4_pi0_compile_modes
```

## Deviations
- **LeRobot e2e.** It is the model call (`predict_action_chunk`, as on the Orin). Preprocess (0.5–0.9 ms) and
  postprocess (0.1 ms) are timed separately in the same iteration (`e2e_full_wall`).
- **GR00T e2e.** It is `Gr00tPolicy.get_action`, which includes its CPU processing (`outside_model`, 14–19 ms).
- **GR00T inputs.** Seeded random pixels at the checkpoint's shapes (as on the Orin), not a real frame.
- **Floors.** From executed-op accounting rather than per-architecture formulas.
- **E4.** At N ∈ {2, 6, 10} (slope from 3 points) and at matmul `highest`.
- **Added beyond the task.** The shipped-path row ran at the default N only. The N1.7 bimodality check was added.

## Self-review (protocol §11)
| check | result |
|---|---|
| six models, E3 at all six N, separate vision and prefix | yes |
| GR00T split validated | yes: disjoint for all; sum within 2 % for N1.6 / N1.5; N1.7 fails with the cause identified (unused `lm_head` logits) |
| `floor.csv` per model against T003's ceilings | yes |
| π0's four stock modes: compile time, graph breaks, ms per step | yes (+ recompiles, CUDA graphs, SM message, peak memory) |
| `meta.json` complete, raw files committed, secrets scan clean, no file > 50 MB | yes (14 directories) |
| sizes in `STATUS.md`, `PROGRESS.md` updated | at the close |
| independent review (P0) | see the T004 status block |
