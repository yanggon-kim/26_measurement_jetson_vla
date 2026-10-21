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
