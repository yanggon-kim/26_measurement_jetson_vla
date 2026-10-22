# T004 — Baseline latency for all six models (E3, N sweep, floors) and π0's stock compile modes (E4) · priority P0 · depends on: T003

## Why
The eager baseline is half of deliverable 1, and its regions are the instrumentation every later task reuses. This
task makes the region split correct for all six models — in particular GR00T's vision vs LLM split, which the Orin
got wrong at first — and computes each part's physical floor against this device's ceilings.

## Orin counterpart
`reference/orin/tasks/T002_pi0_eager.md`, `T003_pi0_compile.md`, `T006_more_models_and_libero.md` step 1,
`T009_groot.md`; reports `reference/orin/reports/E3_pi0_eager.md`, `E4_pi0_compile.md`, `E3_E4_models.md`,
`E3_E4_groot.md`; scripts `reference/orin/bench/regions.py`, `model_registry.py`, `run_e3.py`, `run_e4.py`,
`groot_phases.py`, `floors_models.py`, `floors_groot.py`, `regen_floor.py`.

## Configuration
Maximum-performance mode, `jetson_clocks`, gate passed. Each model in its T001 environment, on its fixture, batch 1.
Eager = the rung `eager_fp32` (`torch.set_float32_matmul_precision("highest")`, the checkpoint's shipped dtypes, TF32
explicitly off) and, as a second row, the model's shipped eager path as-is (record what precision it uses).

## Steps
1. **Port the harness**: `bench/model_registry.py` (six models, N field, wrap points), `bench/regions.py` (protocol
   §2 regions as NVTX ranges + CUDA events, installed after any compilation), `bench/common.py` (gate, meta.json,
   raw-file hashing, compression and the secrets scan of `reference/orin/bench/finalize_raw.py`). Commit before use.
2. **GR00T's vision / LLM split.** Wrap the vision-tower call and the LLM call inside the backbone's forward as
   `vision` and `vlm_prefix` for N1.7, N1.6 and N1.5. Validate per model (protocol §2): the two regions sum to the
   backbone call within 2 % and their kernel sets in a short nsys capture are disjoint; commit the check's output.
3. **Floors** (protocol §9) per model and part (`vision`, `vlm_prefix`, `ae_step`, `ae_total`) from the loaded
   model's parameters and the real P and S, against T003's measured peaks → `floor.csv`.
4. **E3** per model: eager, N ∈ {1, 2, 4, 6, 8, 10}, ≥ 20 warm-up / ≥ 100 timed; per region median / p5 / p95 / n
   and share; ms per step instrumented and by slope (fit a + b·N with residuals); P processed and valid, S; peak
   memory; Hz. π0 first, then π0.5, SmolVLA, GR00T N1.7, N1.6, N1.5.
5. **E4 stock modes (π0)** (plan E4, first bullet): `default`, `reduce-overhead`, `max-autotune`,
   `max-autotune-no-cudagraphs`, expert alone and whole policy; compile time, graph breaks, recompilations, latency,
   ms per step by slope, peak memory, graphs used or not, the Inductor SM-threshold message.

## Result directories and reports
`results/<date>_E3_<model>_eager_Nsweep/` (one per model; `latency_samples.csv`, `latency_summary.csv`,
`step_fit.csv`, `floor.csv`, `raw/`), `results/<date>_E3_groot_region_split_check/`,
`results/<date>_E4_pi0_compile_modes/`; `reports/E3_E4_baseline.md` — one table across models (per region ms and
share at the default N, ms per step both ways, floor and gap ratio per part), the Orin values beside them
(`reference/orin/reports/E3_E4_models.md`, `E3_E4_groot.md`, labelled Orin), and π0's compile-mode table.

## Acceptance
- Six models (or each missing one marked blocked with its error) with E3 at all six N, regions separate for vision
  and prefix, the GR00T split validated, `floor.csv` per model against T003's ceilings.
- π0's four stock compile modes with compile time, graph breaks and ms per step.
- `meta.json` complete; raw files committed; sizes in `STATUS.md`; `PROGRESS.md` updated; independent review (P0).

## Push points
After each model's E3 directory, after the E4 directory, at the close.

## Status (appended by the on-device agent)

status: claimed 2026-10-22
status: done 2026-10-22
- Result directories (current):
  - `results/20261022_E3_{pi0,pi05,smolvla}_eager_Nsweep/` and `results/20261022_E3_{pi0,pi05,smolvla}_shipped_N10/`;
  - `results/20261022_E3_groot_n1{7,6,5}_eager_Nsweep_v2/` and `results/20261022_E3_groot_n1{7,6,5}_shipped_N4_v2/`;
  - split check: `results/20261022_E3_region_split_check/` (LeRobot rows) and
    `results/20261022_E3_groot_region_split_check/`;
  - `results/20261022_E4_pi0_compile_modes/`, `results/20261022_E3_groot_n17_state_vs_latency/`,
    `results/20261022_E3_lerobot_accounting_flop_ops/`.
- Superseded (kept, with `SUPERSEDED.md`): the six GR00T v1 E3 directories, and the GR00T rows of
  `results/20261022_E3_region_split_check/`.
- Report: `reports/E3_E4_baseline.md`.
- bench commits: a32aa34 (E3 LeRobot), c34fd33 (review fixes: regions, accounting, E4 recompile evidence), ce769e5
  (report).
- Headline at the default N (`eager_fp32`), action-expert share of e2e:
  - π0 29.7 %; π0.5 28.2 %; SmolVLA 80.6 %;
  - GR00T N1.7 47.8 %, N1.6 54.2 %, N1.5 37.1 %. Of the model call, the Orin's GR00T definition: 54.7 / 60.9 / 42.3 %.
  - Thor is 2.1–4.3× faster than the Orin in eager mode.
- Independent review (P0): two passes. The first found 11 issues, all fixed with reruns (v2):
  - the GR00T prefix definitions were inconsistent across versions;
  - CategorySpecificLinear bytes were overcounted;
  - restream was compared with the wrong quantity;
  - the E4 recompile counter was dead;
  - the N1.7 bimodality claim was weak;
  - meta nulls.
  The second checked the rewritten report number by number and found:
  - a wrong speed-up claim (true for π0 only);
  - the N1.7 share range;
  - rounding;
  - the directory count;
  - missing P valid, S, peak memory and Hz.
  All fixed. While porting T005 I also found that the Orin's GR00T e2e is the model call (GPU path); the report now
  compares GR00T on that basis.
- Deviations:
  - GR00T e2e is `Gr00tPolicy.get_action`; `model_get_action` is reported beside it.
  - GR00T `vlm_prefix` is the decoder stack, with the unused `lm_head` as its own region. Thor's prefix is therefore not
    the Orin's (backbone − vision); both are reported.
  - The N1.7 split sums to 0.952, not within 2 %; the remaining ≈ 2.9 ms is identified (CPU-side `get_rope_index`,
    0.25 ms glue kernels).
  - The nsys split check ran for `eager_fp32` only.
  - E4 ran at N ∈ {2, 6, 10}.
  - N1.7's expert time varies 72.2–85.7 ms between processes; the cause is open and goes to T006.
